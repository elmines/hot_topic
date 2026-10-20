#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
from collections import Counter
import logging
from multiprocessing import Pool, cpu_count
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from pathlib import Path
from uuid import uuid4

from openai import AsyncOpenAI
from tqdm import tqdm
from vllm.config import DeviceConfig, ModelConfig, VllmConfig
from vllm.exceptions import VLLMValidationError
from vllm.renderers import ChatParams, BaseRenderer, renderer_from_config

logger = logging.getLogger(__name__)

MAX_LINES_PER_REQUEST = 300
CHUNK_OVERLAP = 24
REASONING_EFFORT_CHOICES = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

AD_SPAN_SCHEMA = {
    "type": "object",
    "properties": {
        "spans": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "integer", "minimum": 0},
                    "end": {"type": "integer", "minimum": 0},
                    "reason": {"type": "string"},
                },
                "required": ["start", "end"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["spans"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """You identify advertising in podcast transcripts.

Return one JSON object matching the provided schema, with a `spans` array. Each span must contain integer `start` and `end` fields and may contain a short `reason` field.
Spans use closed intervals: `start` and `end` are both inclusive. Indices refer to the 0-based line numbers supplied in the transcript.
"""

USER_PROMPT = """Read the transcript below and identify every contiguous range of transcript lines that is advertising. Include sponsor reads, product or service promotions, affiliate messages, fundraising or membership appeals, and promotions for the podcast or its network. Do not label ordinary discussion of a product, news, or a guest's work unless it is promotional.

Return JSON matching the schema. Use closed spans: `start` is the first advertising line and `end` is the last advertising line. For example, {"start": 0, "end": 2} labels lines 0, 1, and 2.

Use the supplied line indices exactly. If there is no advertising, return {"spans": []}. Do not include any text outside the JSON object.

TRANSCRIPT:
"""

_SPAN_FIELDS = (
    r'(?=[^{}]*"start"\s*:\s*(?P<start>\d+))'
    r'(?=[^{}]*"end"\s*:\s*(?P<end>\d+))'
    r'(?=[^{}]*"reason"\s*:\s*"(?P<reason>(?:\\.|[^"\\])*?)")?'
)
_SPAN_PATTERN = re.compile(r"\{" + _SPAN_FIELDS + r"[^{}]*\}")
_ORPHAN_SPAN_PATTERN = re.compile(
    r"\{"
    r'(?=[^{}]*"start"\s*:\s*(?P<start>\d+))'
    r'(?=[^{}]*"end"\s*:\s*(?P<end>\d+))'
    r'(?=[^{}]*"reason"\s*:\s*"(?P<reason>(?:\\.|[^"\\])*?)")?'
    r"[^{}]*$"
)


def read_transcript(path):
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
    print(f"{path.name}: Warning: {message}", file=sys.stderr)


def extract_json_object(text, path, extract_answer):
    text = extract_answer(text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "spans" in value:
            return value

    spans = [
        {
            "start": int(match.group("start")),
            "end": int(match.group("end")),
            "reason": (
                json.loads(f'"{match.group("reason")}"')
                if match.group("reason") is not None
                else ""
            ),
        }
        for match in _SPAN_PATTERN.finditer(text)
    ]

    orphan_match = _ORPHAN_SPAN_PATTERN.search(text)
    if orphan_match:
        spans.append({
            "start": int(orphan_match.group("start")),
            "end": int(orphan_match.group("end")),
            "reason": (
                json.loads(f'"{orphan_match.group("reason")}"')
                if orphan_match.group("reason") is not None
                else ""
            ),
        })
    if spans:
        return {"spans": spans}

    warning(path, f"LLM response did not contain a spans JSON object: {text[:500]!r}")
    return {"spans": []}


def validate_spans(payload, line_count, offset, path):
    if not isinstance(payload, dict) or not isinstance(payload.get("spans"), list):
        warning(path, "LLM response must contain a spans array")
        return []
    validated = []
    lower = 0
    upper = line_count
    for span in payload["spans"]:
        if not isinstance(span, dict) or not isinstance(span.get("start"), int) or not isinstance(span.get("end"), int):
            warning(path, f"Malformed span: {span!r}")
            continue
        start, end = span["start"], span["end"]
        if not lower <= start <= end:
            warning(path, f"Span {span!r} has an invalid closed line range")
            continue
        if end >= upper:
            warning(path, f"Span {span!r} extends past the chunk; truncating its end to {upper - 1}")
            end = upper - 1
        if end < start:
            warning(path, f"Span {span!r} has no lines in the chunk")
            continue
        validated.append({"start": start + offset, "end": end + offset, "reason": str(span.get("reason", ""))})
    return validated


def merge_spans(spans):
    merged = []
    for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
        if merged and span["start"] <= merged[-1]["end"] + 1:
            merged[-1]["end"] = max(merged[-1]["end"], span["end"])
            if span.get("reason") and span["reason"] not in merged[-1].get("reason", ""):
                merged[-1]["reason"] = f"{merged[-1].get('reason', '')}; {span['reason']}".strip("; ")
        else:
            merged.append(dict(span))
    return merged


def render_chunk(renderer, offset, chunk):
    return renderer.render_chat(
        [[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + format_transcript(chunk)},
        ]],
        ChatParams(chat_template_kwargs={"add_generation_prompt": True}),
    )[1][0]


def prompt_token_count(rendered_prompt):
    try:
        return len(rendered_prompt["prompt_token_ids"])
    except (KeyError, TypeError):
        return len(rendered_prompt.prompt_token_ids)


def try_render_chunk(renderer, offset, chunk):
    try:
        return render_chunk(renderer, offset, chunk)
    except VLLMValidationError:
        return None


def truncate_chunk_line(renderer, offset, chunk, max_prompt_tokens, path):
    row = chunk[0]
    text = row["text"]
    low, high = 0, len(text)
    best = None
    while low <= high:
        middle = (low + high) // 2
        candidate = [dict(row, text=text[:middle])]
        rendered_prompt = try_render_chunk(renderer, offset, candidate)
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


def transcript_chunks(rows, renderer, max_model_len, max_new_tokens, path):
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
        while line_count > 1:
            chunk = rows[offset:offset + line_count]
            rendered_prompt = try_render_chunk(renderer, offset, chunk)
            if rendered_prompt is not None and prompt_token_count(rendered_prompt) <= max_prompt_tokens:
                break
            line_count -= 1
        else:
            chunk = rows[offset:offset + 1]
            rendered_prompt = try_render_chunk(renderer, offset, chunk)
            if rendered_prompt is None or prompt_token_count(rendered_prompt) > max_prompt_tokens:
                chunk, rendered_prompt = truncate_chunk_line(
                    renderer, offset, chunk, max_prompt_tokens, path
                )

        chunks.append((offset, chunk, rendered_prompt))
        offset += max(1, len(chunk) - CHUNK_OVERLAP)
    return chunks


async def generate_text(
    client: AsyncOpenAI,
    model,
    chunk,
    max_new_tokens,
    reasoning_effort,
    request_id,
):
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT + format_transcript(chunk)},
        ],
        temperature=0.0,
        max_tokens=max_new_tokens,
        reasoning_effort=reasoning_effort,
    )
    if not response.choices or response.choices[0].message.content is None:
        warning(Path(request_id), "LLM response was empty")
        return ""
    return response.choices[0].message.content


async def generate_chunk_responses(
    client, model, offset, chunk, max_new_tokens, reasoning_effort, transcript_path
):
    response = await generate_text(
        client,
        model,
        chunk,
        max_new_tokens,
        reasoning_effort,
        f"chunk-{offset}-{uuid4()}",
    )
    if response.strip() or len(chunk) < 2:
        return [(offset, chunk, response)]

    midpoint = len(chunk) // 2
    logger.warning(
        "Splitting empty-response chunk for transcript %s at line %d",
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
        ),
        generate_chunk_responses(
            client,
            model,
            offset + midpoint,
            chunk[midpoint:],
            max_new_tokens,
            reasoning_effort,
            transcript_path,
        ),
    )
    return left + right

async def label_transcript(
    client, model, renderer: BaseRenderer, input_path, max_new_tokens,
    reasoning_effort, extract_answer, max_model_len,
):
    rows = read_transcript(input_path)
    all_spans = []
    chunks = transcript_chunks(
        rows, renderer, max_model_len, max_new_tokens, input_path
    ) if rows else []
    responses = await asyncio.gather(*(
        generate_chunk_responses(
            client,
            model,
            offset,
            chunk,
            max_new_tokens,
            reasoning_effort,
            input_path,
        )
        for offset, chunk, _ in chunks
    ))
    for offset, chunk, response in (
        result
        for chunk_results in responses
        for result in chunk_results
    ):
        payload = extract_json_object(response, input_path, extract_answer)
        all_spans.extend(validate_spans(payload, len(chunk), offset, input_path))
    return {
        "source": str(input_path),
        "line_count": len(rows),
        "spans": merge_spans(all_spans),
    }


def input_paths(root, inputs):
    root = root.resolve()
    paths = []
    for input_path in inputs:
        path = input_path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"--input file is outside --root: {input_path}")
        if not path.is_file():
            raise ValueError(f"--input file does not exist: {path}")
        paths.append(path)
    return paths


def output_path(root, output_dir, input_path):
    relative = input_path.relative_to(root)
    return (output_dir / relative).with_suffix(".json")


def make_renderer(args):
    model_config = ModelConfig(
        model=args.model,
        max_model_len=args.max_model_len,
    )
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
    )
    return renderer_from_config(config)


def server_request(base_url, path, api_key):
    request = Request(
        f"{base_url.rstrip('/')}/{path.lstrip('/')}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(request) as response:
            return json.load(response)
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"could not request {request.full_url}: {exc}") from exc


def server_model(base_url, api_key):
    response = server_request(base_url, "/v1/models", api_key)
    try:
        model = response["data"][0]["id"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("vLLM /v1/models response did not contain a model") from exc
    return model


def server_max_model_len(base_url, api_key):
    response = server_request(base_url, "/server_info", api_key)
    for key in ("max_model_len", "max_seq_len"):
        value = response.get(key) if isinstance(response, dict) else None
        if isinstance(value, int):
            return value

    config = response.get("vllm_config") if isinstance(response, dict) else None
    match = re.search(r"(?:max_model_len|max_seq_len)=(\d+)", str(config))
    if match:
        return int(match.group(1))
    raise RuntimeError("vLLM /server_info response did not contain max_model_len")


_ESTIMATE_RENDERER = None
_ESTIMATE_MAX_MODEL_LEN = None
_ESTIMATE_MAX_NEW_TOKENS = None


def initialize_estimator(model, max_model_len, max_new_tokens):
    global _ESTIMATE_RENDERER, _ESTIMATE_MAX_MODEL_LEN, _ESTIMATE_MAX_NEW_TOKENS
    _ESTIMATE_RENDERER = make_renderer(
        argparse.Namespace(model=model, max_model_len=max_model_len)
    )
    _ESTIMATE_MAX_MODEL_LEN = max_model_len
    _ESTIMATE_MAX_NEW_TOKENS = max_new_tokens


def estimate_file_token_lengths(input_path):
    rows = read_transcript(input_path)
    chunks = transcript_chunks(
        rows,
        _ESTIMATE_RENDERER,
        _ESTIMATE_MAX_MODEL_LEN,
        _ESTIMATE_MAX_NEW_TOKENS,
        input_path,
    ) if rows else []
    frequencies = Counter()
    maximum = 0
    for _, _, rendered_prompt in chunks:
        token_count = prompt_token_count(rendered_prompt) + _ESTIMATE_MAX_NEW_TOKENS
        frequencies[token_count // 5000 * 5000] += 1
        maximum = max(maximum, token_count)
    return frequencies, maximum


def estimate_token_lengths(args, input_files):
    frequencies = Counter()
    maximum = 0
    worker_count = min(args.num_workers, len(input_files))
    with Pool(
        processes=worker_count,
        initializer=initialize_estimator,
        initargs=(args.model, args.max_model_len, args.max_new_tokens),
    ) as pool:
        for file_frequencies, file_maximum in tqdm(
            pool.imap_unordered(estimate_file_token_lengths, input_files),
            total=len(input_files),
            desc="Counting tokens from chunks in input files",
        ):
            frequencies.update(file_frequencies)
            maximum = max(maximum, file_maximum)

    writer = csv.writer(sys.stdout)
    writer.writerow(("token_length", "count"))
    bucket_starts = sorted(frequencies)
    for bucket_start in bucket_starts[:-1]:
        writer.writerow((f"{bucket_start}-{bucket_start + 5000}", frequencies[bucket_start]))
    if bucket_starts:
        bucket_start = bucket_starts[-1]
        writer.writerow((f"{bucket_start}-{maximum}", frequencies[bucket_start]))


async def run(args, input_files, output_dir):
    client = AsyncOpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key=args.api_key,
    )
    model = args.model
    renderer = make_renderer(
        argparse.Namespace(model=model, max_model_len=args.max_model_len)
    )
    extract_answer = lambda text: text

    async def process(path):
        destination = output_path(args.root.resolve(), output_dir, path)
        if destination.exists():
            print(f"Skipping transcript {path}: output already exists at {destination}")
            return 0
        result = await label_transcript(
            client,
            model,
            renderer,
            path,
            args.max_new_tokens,
            args.reasoning_effort,
            extract_answer,
            args.max_model_len,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return 1

    n_transcribed = 0
    duration = -time.time()
    tasks = [asyncio.create_task(process(path)) for path in input_files]
    with tqdm(total=len(tasks), desc="Writing transcripts") as progress:
        for task in asyncio.as_completed(tasks):
            n_transcribed += await task
            progress.update(1)
    duration += time.time()

    print("n_transcribed", "duration", sep=',')
    print(n_transcribed, duration, sep=',')


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Extract advertising spans from transcript CSVs")
    parser.add_argument("-i", "--input", dest="inputs", nargs="+", required=True, type=Path,
                        help="Transcript CSV files under --root")
    parser.add_argument("--root", default=Path("out/whisper_segmented"), type=Path,
                        help="Root directory under which every --input file must be found")
    parser.add_argument("-o", "--output", default=Path("out/ad_spans"), type=Path,
                        help="Directory for JSON outputs, mirroring --root")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key", default="derf")
    parser.add_argument("--max-new-tokens", type=int, default=1 << 11)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORT_CHOICES,
        default="low",
        help="Reasoning effort passed to the vLLM chat completion",
    )
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
    args = parser.parse_args(raw_args)

    if args.max_new_tokens < 1 or args.num_workers < 1 or not 1 <= args.port <= 65535:
        parser.error("token count, worker count, and port must be valid positive values")

    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"--root is not a directory: {root}")
    output_dir = args.output.resolve()
    if output_dir == root or output_dir.is_relative_to(root) or root.is_relative_to(output_dir):
        parser.error("--output must be a different directory from --root")
    try:
        input_files = input_paths(root, args.inputs)
    except ValueError as exc:
        parser.error(str(exc))

    args.max_model_len = server_max_model_len(
        f"http://{args.host}:{args.port}", args.api_key
    )
    args.model = server_model(
        f"http://{args.host}:{args.port}", args.api_key
    )
    if args.max_model_len <= args.max_new_tokens:
        parser.error("server max_model_len must be greater than --max-new-tokens")

    if args.estimate_tokens:
        estimate_token_lengths(args, input_files)
    else:
        asyncio.run(run(args, input_files, output_dir))


if __name__ == "__main__":
    main()