import asyncio
import csv
import json
import logging
import re
import sys
import time
import tomllib
from multiprocessing import cpu_count
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import uuid4

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers import ChatParams, renderer_from_config

logger = logging.getLogger(__name__)

MAX_LINES_PER_REQUEST = 300
CHUNK_OVERLAP = 24
REQUEST_CONCURRENCY = 64
REASONING_EFFORT_CHOICES = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

VLLM_DEFAULTS = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "vllm_defaults.cfg").read_text(
        encoding="utf-8"
    )
)


def add_vllm_client_arguments(parser):
    parser.add_argument("--base-url", default="http://0.0.0.0:8001/v1")
    parser.add_argument("--api-key", default="derf")
    parser.add_argument(
        "--model",
        default=None,
        help="Model to use when the server exposes multiple models",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1 << 11)
    parser.add_argument("--max-model-len", type=int, default=VLLM_DEFAULTS["HT_MAX_MODEL_LEN"])
    parser.add_argument("--max-num-seqs", type=int, default=VLLM_DEFAULTS["HT_MAX_NUM_SEQS"])
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=VLLM_DEFAULTS["HT_MAX_NUM_BATCHED_TOKENS"],
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default=None,
        help="Reasoning effort passed to the vLLM chat completion",
    )
    parser.add_argument(
        "--episode-batch",
        type=int,
        default=8,
        help="Number of episodes to process concurrently before starting the next batch",
    )


def add_token_estimate_arguments(parser):
    parser.add_argument(
        "--num-workers",
        type=int,
        default=min(1, cpu_count() - 1),
        help="Worker processes for CPU-side transcript preparation",
    )
    parser.add_argument(
        "--estimate-tokens",
        action="store_true",
        help="Count prompt lengths without creating an LLM engine",
    )


def validate_vllm_arguments(parser, args):
    num_workers = getattr(args, "num_workers", 1)
    if (
        args.max_new_tokens < 1
        or args.max_model_len < 1
        or args.max_num_seqs < 1
        or args.max_num_batched_tokens < 1
        or num_workers < 1
        or args.episode_batch < 1
    ):
        parser.error(
            "token counts, worker count, and episode batch "
            "must be valid positive values"
        )


def configure_server(parser, args):
    try:
        models = server_models(args.base_url, args.api_key)
    except RuntimeError as exc:
        parser.error(str(exc))
    if args.model is None:
        if len(models) != 1:
            parser.error("server exposes multiple models; specify --model")
        args.model = models[0]
    elif args.model not in models:
        parser.error(f"--model {args.model!r} is not exposed by the server")
    if args.max_model_len <= args.max_new_tokens:
        parser.error("server max_model_len must be greater than --max-new-tokens")


def make_openai_client(args):
    return AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=900.0,
        max_retries=0,
    )


def make_renderer(model, max_model_len):
    model_config = ModelConfig(
        model=model,
        max_model_len=max_model_len,
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
    )
    return renderer_from_config(config)


def server_request(base_url, path, api_key, *, attempts=30, retry_delay=10):
    request = Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    last_error = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=10) as response:
                return json.load(response)
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504}:
                break
        except (URLError, TimeoutError) as exc:
            last_error = exc
        if attempt + 1 < attempts:
            time.sleep(retry_delay)
    raise RuntimeError(
        f"could not request {request.full_url} after {attempts} attempts: {last_error}"
    ) from last_error


def server_models(base_url, api_key):
    response = server_request(base_url, "/models", api_key)
    try:
        models = [model["id"] for model in response["data"]]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("models endpoint response did not contain models") from exc
    if not models or not all(isinstance(model, str) and model for model in models):
        raise RuntimeError("models endpoint response did not contain valid models")
    return models


def read_transcript(path):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    if reader.fieldnames:
        missing = {"start", "end", "text"} - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def format_transcript(rows):
    return "\n".join(f"[{index}] {row['text'].strip()}" for index, row in enumerate(rows))


def warning(path, message):
    print(f"{Path(path).name}: Warning: {message}", file=sys.stderr)


def parse_json_object(text, extract_answer=lambda text: text):
    text = extract_answer(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def payload_with_array(payload, key):
    if payload is None:
        return None
    if key not in payload:
        return {key: []}
    return payload


def prompt_token_count(rendered_prompt):
    try:
        return len(rendered_prompt["prompt_token_ids"])
    except (KeyError, TypeError):
        return len(rendered_prompt.prompt_token_ids)


def render_chat(renderer, messages):
    return renderer.render_chat(
        [messages],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )[1][0]


def try_render_chat(renderer, messages):
    try:
        return render_chat(renderer, messages)
    except VLLMValidationError:
        return None


def truncate_chunk_line(renderer, messages_for_chunk, chunk, max_prompt_tokens, path):
    row = chunk[0]
    text = row["text"]
    low, high = 0, len(text)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = [dict(row, text=text[:middle])]
        rendered_prompt = try_render_chat(renderer, messages_for_chunk(candidate))
        if rendered_prompt is not None and prompt_token_count(rendered_prompt) <= max_prompt_tokens:
            best = (candidate, rendered_prompt)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError(
            f"cannot fit even an empty transcript line in the prompt budget for {path}"
        )
    warning(path, "truncated a transcript line to fit the prompt token budget")
    return best


def transcript_chunks(rows, renderer, max_model_len, max_new_tokens, path, messages_for_chunk):
    renderer_model_len = getattr(getattr(renderer, "model_config", None), "max_model_len", None)
    if renderer_model_len is not None:
        max_model_len = min(max_model_len, renderer_model_len)
    max_prompt_tokens = max_model_len - max_new_tokens
    if max_prompt_tokens < 1:
        raise ValueError("max_model_len must be greater than max_new_tokens")

    chunks = []
    offset = 0
    while offset < len(rows):
        line_count = min(MAX_LINES_PER_REQUEST, len(rows) - offset)
        print(f"path={path}, offset={offset}, line_count={line_count}")
        while line_count > 1:
            chunk = rows[offset:offset + line_count]
            rendered_prompt = try_render_chat(renderer, messages_for_chunk(chunk))
            if rendered_prompt is not None and prompt_token_count(rendered_prompt) <= max_prompt_tokens:
                break
            line_count -= 1
        else:
            chunk = rows[offset:offset + 1]
            rendered_prompt = try_render_chat(renderer, messages_for_chunk(chunk))
            if rendered_prompt is None or prompt_token_count(rendered_prompt) > max_prompt_tokens:
                chunk, rendered_prompt = truncate_chunk_line(
                    renderer, messages_for_chunk, chunk, max_prompt_tokens, path
                )

        chunks.append((offset, chunk, rendered_prompt))
        if offset + line_count >= len(rows):
            # Without this, we'll generate several overlapping chunks at the end of the transcript
            break

        offset += max(1, len(chunk) - CHUNK_OVERLAP)
    return chunks


async def generate_text(
    client,
    model,
    messages,
    max_new_tokens,
    reasoning_effort,
    request_id,
    request_semaphore=None,
):
    request = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_new_tokens
    }
    if reasoning_effort is not None:
        request['reasoning_effort'] = reasoning_effort

    for attempt in range(6):
        try:
            if request_semaphore is None:
                response = await client.chat.completions.create(**request)
            else:
                async with request_semaphore:
                    response = await client.chat.completions.create(**request)
            break
        except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError):
            if attempt == 5:
                raise
            await asyncio.sleep(min(60, 2 ** attempt))
    if not response.choices or response.choices[0].message.content is None:
        warning(Path(request_id), "LLM response was empty")
        return ""
    return response.choices[0].message.content


async def generate_chunk_responses(
    client,
    model,
    offset,
    chunk,
    max_new_tokens,
    reasoning_effort,
    transcript_path,
    messages_for_chunk,
    parse_payload,
    request_semaphore=None,
    prompt_debug_path=None,
):
    request_id = f"chunk-{offset}-{uuid4()}"
    messages = messages_for_chunk(chunk)
    if prompt_debug_path is not None:
        with Path(prompt_debug_path).open("w", encoding="utf-8") as debug_file:
            debug_file.write(messages[-1]["content"])
    if request_semaphore is None:
        response = await generate_text(
            client,
            model,
            messages,
            max_new_tokens,
            reasoning_effort,
            request_id,
        )
    else:
        response = await generate_text(
            client,
            model,
            messages,
            max_new_tokens,
            reasoning_effort,
            request_id,
            request_semaphore=request_semaphore,
        )
    response = response.strip()
    payload = parse_payload(response)

    if payload is not None:
        return [(offset, chunk, payload)]

    split_reason = "empty-response" if not response else "invalid-JSON"
    if len(chunk) < 2:
        logger.warning(
            "%s for 1-line chunk from transcript %s at line %d",
            split_reason,
            transcript_path,
            offset,
        )
        return []

    midpoint = len(chunk) // 2
    logger.warning(
        "Splitting %s chunk for transcript %s at line %d",
        split_reason,
        transcript_path,
        offset + midpoint,
    )
    left, right = await asyncio.gather(
        generate_chunk_responses(
            client,
            model,
            offset,
            chunk[:midpoint],
            max_new_tokens,
            reasoning_effort,
            transcript_path,
            messages_for_chunk,
            parse_payload,
            request_semaphore,
            prompt_debug_path,
        ),
        generate_chunk_responses(
            client,
            model,
            offset + midpoint,
            chunk[midpoint:],
            max_new_tokens,
            reasoning_effort,
            transcript_path,
            messages_for_chunk,
            parse_payload,
            request_semaphore,
            prompt_debug_path,
        ),
    )
    return left + right


def episode_batches(input_files, episode_batch):
    return [
        input_files[start:start + episode_batch]
        for start in range(0, len(input_files), episode_batch)
    ]
