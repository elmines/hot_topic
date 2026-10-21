#!/usr/bin/env python3

import argparse
import asyncio
import csv
import json
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from hot_topic.llm_utils import (
    REQUEST_CONCURRENCY,
    add_token_estimate_arguments,
    add_vllm_client_arguments,
    configure_vllm_server,
    episode_batches,
    format_transcript,
    generate_chunk_responses,
    make_openai_client,
    make_renderer,
    parse_json_object as parse_json_value,
    payload_with_array,
    prompt_token_count,
    read_transcript,
    transcript_chunks,
    validate_vllm_arguments,
    warning,
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


def parse_json_object(text, extract_answer):
    return payload_with_array(parse_json_value(text, extract_answer), "spans")


def span_messages(chunk):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT + format_transcript(chunk)},
    ]


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


async def label_transcript(
    client, model, renderer, input_path, max_new_tokens,
    reasoning_effort, extract_answer, max_model_len, request_semaphore=None,
):
    rows = read_transcript(input_path)

    def parse_payload(text):
        return parse_json_object(text, extract_answer)

    all_spans = []
    chunks = transcript_chunks(
        rows, renderer, max_model_len, max_new_tokens, input_path, span_messages
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
            span_messages,
            parse_payload,
            request_semaphore,
        )
        for offset, chunk, _rendered in chunks
    ))
    for offset, chunk, payload in (
        result
        for chunk_results in responses
        for result in chunk_results
    ):
        all_spans.extend(validate_spans(payload, len(chunk), offset, input_path))
    return {
        "source": str(input_path),
        "line_count": len(rows),
        "spans": merge_spans(all_spans),
    }


_ESTIMATE_RENDERER = None
_ESTIMATE_MAX_MODEL_LEN = None
_ESTIMATE_MAX_NEW_TOKENS = None


def initialize_estimator(model, max_model_len, max_new_tokens):
    global _ESTIMATE_RENDERER, _ESTIMATE_MAX_MODEL_LEN, _ESTIMATE_MAX_NEW_TOKENS
    _ESTIMATE_RENDERER = make_renderer(model, max_model_len)
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
        span_messages,
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
    client = make_openai_client(args)
    request_semaphore = asyncio.Semaphore(REQUEST_CONCURRENCY)
    model = args.model
    renderer = make_renderer(model, args.max_model_len)
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
            request_semaphore,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
            stream.write("\n")
        return 1

    n_transcribed = 0
    duration = -time.time()
    with tqdm(total=len(input_files), desc="Writing transcripts") as progress:
        for batch_files in episode_batches(input_files, args.episode_batch):
            tasks = [asyncio.create_task(process(path)) for path in batch_files]
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
    add_vllm_client_arguments(parser)
    add_token_estimate_arguments(parser)
    args = parser.parse_args(raw_args)
    validate_vllm_arguments(parser, args)

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

    configure_vllm_server(parser, args)

    if args.estimate_tokens:
        estimate_token_lengths(args, input_files)
    else:
        asyncio.run(run(args, input_files, output_dir))


if __name__ == "__main__":
    main()
