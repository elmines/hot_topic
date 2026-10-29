#!/usr/bin/env python3

import argparse
import asyncio
import csv
import glob
import json
import os
from collections import Counter, OrderedDict
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .constants import DEFAULT_TOPICS, NOISE_PROMPT
from .csv_writer import get_csv_writer
from .llm_utils import (
    REQUEST_CONCURRENCY,
    add_vllm_client_arguments,
    configure_server,
    episode_batches,
    format_transcript,
    generate_chunk_responses,
    make_openai_client,
    make_renderer,
    parse_json_object,
    payload_with_array,
    read_transcript,
    transcript_chunks,
    validate_vllm_arguments,
    warning,
)
from .utils import align_quote, combine_rows, normalize_model_name, preprocess

CHECKPOINT_FILENAME = "processed_topics.json"
RNG = np.random.default_rng(0)

TOPIC_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["topics"],
    "properties": {
        "topics": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["topic", "description", "quote"],
                "properties": {
                    "topic": {"type": "string", "minLength": 1, "maxLength": 30},
                    "description": {"type": "string", "minLength": 1, "maxLength": 256},
                    "quote": {"type": "string", "minLength": 1, "maxLength": 256},
                },
            },
        },
    },
}

SYSTEM_PROMPT = (
    "You identify generalizable topics in podcast transcripts.\n"
    "\n"
    "Return one JSON object and no other text. The object must match this schema:\n"
    f"{json.dumps(TOPIC_SCHEMA, indent=2)}\n"
    "\n"
    "Use `topic` for a short general label, `description` for what that topic covers, "
    "and `quote` for a verbatim excerpt from the transcript with no added quotation marks.\n"
    "Do not repeat the same topic. If the transcript contains no relevant topic, "
    'return {"topics": []}.\n'
)

USER_PROMPT = """
You will receive a document and a set of top-level topics from a topic hierarchy.
Your task is to identify generalizable topics within the document that can act as top-level topics in the hierarchy.
If any relevant topics are missing from the provided set, please add them.
Otherwise, output the existing top-level topics as identified in the document.
For each topic, you must include a quote from the document justifying your choice.

[Top-level topics]
{Topics}

[Examples]
Example 1: Adding "agriculture"
Document: 
Saving Essential American Sailors Act or SEAS Act - Amends the Moving Ahead for Progress in the 21st Century Act (MAP-21) to repeal the Act's repeal of the agricultural export requirements that: (1) 25% of the gross tonnage of certain agricultural commodities or their products exported each fiscal year be transported on U.S. commercial vessels, and (2) the Secretary of Transportation (DOT) finance any increased ocean freight charges incurred in the transportation of such items. Revives and reinstates those repealed requirements to read as if they were never repealed.

Your response: 
{"topics": [{"topic": "agriculture", "description": "mentions policies relating to agricultural practices and products", "quote": "repeal of the agricultural export requirements"}]}

Example 2: Duplicate "trade", returning the existing topic
Document: 
Amends the Harmonized Tariff Schedule of the United States to suspend temporarily the duty on mixtures containing Fluopyram.

Your response: 
{"topics": [{"topic": "trade", "description": "mentions the exchange of capital, goods, and services", "quote": "duty on mixtures containing"}]}

Do not quote noise such as music, promotional messages, public service announcements, or advertisements as justification for topics.
Don't even quote text near such noise, as it may be noise as well.
Here are some examples of such noise:
{Noise}

[Instructions]
Step 1: Determine topics mentioned in the document. 
- The topic labels must be as GENERALIZABLE as possible. They must not be document-specific.
- The topics must reflect a SINGLE topic instead of a combination of topics.
- The new topics must have a level number, a short general label, and a topic description. 
- The topics must be broad enough to accommodate future subtopics. 
Step 2: Perform ONE of the following operations: 
1. If there are already duplicates or relevant topics in the hierarchy, output those topics and stop here. 
2. If the document contains no topic, return "None". 
3. Otherwise, add your topic as a top-level topic. Stop here and output the added topic(s). DO NOT add any additional levels.


[Document]
{Document}

Please ONLY return the relevant or modified topics at the top level in the hierarchy. Your response should be in the following format:
{"topics": [{"topic": "Topic Label", "description": "Topic Description", "quote": "Document Quote"}]}

Your response:"""


def format_topic(topic_name, topic_desc):
    return f"[1] {topic_name} : {topic_desc}"


def topic_prompt(topic_to_desc, topic_frequencies=None, max_topics=100):
    topic_frequencies = topic_frequencies or {}
    topics = list(topic_to_desc)
    if not topics or max_topics < 1:
        return ""
    weights = np.maximum(
        [topic_frequencies.get(topic, 0) for topic in topics],
        1,
    ).astype(float)
    weights /= weights.sum()
    topics = RNG.choice(
        topics,
        size=min(max_topics, len(topics)),
        replace=False,
        p=weights,
    )
    return "\n".join(format_topic(name, topic_to_desc[name]) for name in topics)


def quote_topic_frequencies(quote_path):
    with open(quote_path, newline="", encoding="utf-8") as handle:
        return Counter(
            row["topic"].strip().lower()
            for row in csv.DictReader(handle)
            if row.get("topic") and row["topic"].strip()
        )


def messages_for_chunk(chunk, topic_str):
    user = (
        USER_PROMPT
        .replace("{Topics}", topic_str)
        .replace("{Noise}", NOISE_PROMPT)
        .replace("{Document}", format_transcript(chunk))
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def ad_span_file(input_root, spans_root, transcript_path):
    transcript_path = Path(transcript_path).resolve()
    relative = transcript_path.relative_to(Path(input_root).resolve())
    return Path(spans_root).resolve() / relative.with_suffix(".json")


def load_ad_spans(transcript_path, input_root, spans_root, line_count):
    transcript_path = Path(transcript_path)
    span_path = ad_span_file(input_root, spans_root, transcript_path)
    if not span_path.is_file():
        warning(transcript_path, f"no ad spans at {span_path}; filtering with regexes only")
        return []
    try:
        with span_path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except json.JSONDecodeError:
        warning(transcript_path, f"could not parse ad spans at {span_path}; filtering with regexes only")
        return []
    if not isinstance(payload, dict):
        warning(transcript_path, f"ad span file {span_path} is not a JSON object; filtering with regexes only")
        return []
    recorded = payload.get("line_count")
    if isinstance(recorded, int) and recorded != line_count:
        warning(
            transcript_path,
            f"ad span line_count {recorded} does not match transcript length {line_count}",
        )
    spans = payload.get("spans", [])
    if not isinstance(spans, list):
        warning(transcript_path, f"ad span file {span_path} is missing a spans array")
        return []
    return spans


def drop_ad_span_lines(rows, spans, path=None):
    rows = list(rows)
    covered = set()
    line_count = len(rows)
    for span in spans:
        if (
            not isinstance(span, dict)
            or not isinstance(span.get("start"), int)
            or not isinstance(span.get("end"), int)
        ):
            if path is not None:
                warning(path, f"Malformed ad span: {span!r}")
            continue
        start, end = span["start"], span["end"]
        if end < start:
            if path is not None:
                warning(path, f"Ad span {span!r} has an invalid closed line range")
            continue
        start = max(start, 0)
        end = min(end, line_count - 1)
        if start <= end:
            covered.update(range(start, end + 1))
    return [row for index, row in enumerate(rows) if index not in covered]


def prepare_transcript(rows, spans, path=None):
    return list(preprocess(drop_ad_span_lines(rows, spans, path)))


def parse_topics_payload(text, extract_answer=lambda text: text):
    return payload_with_array(parse_json_object(text, extract_answer), "topics")


def topic_mentions(payload, path):
    if not isinstance(payload, dict) or not isinstance(payload.get("topics"), list):
        warning(path, "LLM response must contain a topics array")
        return []
    mentions = []
    for item in payload["topics"]:
        if not isinstance(item, dict):
            warning(path, f"Malformed topic: {item!r}")
            continue
        topic = item.get("topic")
        description = item.get("description")
        quote = item.get("quote")
        if not all(isinstance(value, str) and value.strip() for value in (topic, description, quote)):
            warning(path, f"Malformed topic: {item!r}")
            continue
        mentions.append((topic.strip().lower(), description.strip(), quote.strip()))
    return mentions


def select_episode_topics(mentions, text, minimum_overlap):
    chosen = OrderedDict()
    for topic, description, raw_quote in mentions:
        if topic in chosen:
            continue
        quote = align_quote(raw_quote, text, minimum_overlap)
        if quote:
            chosen[topic] = (description, quote)
    return [(topic, description, quote) for topic, (description, quote) in chosen.items()]


def incorporate_topics(topic_to_desc, episode_results):
    new_topics = []
    quote_rows = []
    for path, matches in episode_results:
        for topic, description, quote in matches:
            if topic not in topic_to_desc:
                topic_to_desc[topic] = description
                new_topics.append([topic, description])
            quote_rows.append([path, topic, quote])
    return new_topics, quote_rows


async def discover_batch(
    process,
    topic_to_desc,
    topic_str,
    batch,
    topic_frequencies=None,
    max_prompt_topics=100,
):
    results = await asyncio.gather(*(process(path, topic_str) for path in batch))
    new_topics, quote_rows = incorporate_topics(topic_to_desc, results)
    if topic_frequencies is None:
        topic_frequencies = Counter()
    topic_frequencies.update(topic for _path, topic, _quote in quote_rows)
    return (
        topic_prompt(topic_to_desc, topic_frequencies, max_prompt_topics),
        new_topics,
        quote_rows,
    )


async def process_episode(
    client,
    model,
    renderer,
    path,
    topic_str,
    max_new_tokens,
    reasoning_effort,
    max_model_len,
    input_root,
    spans_root,
    minimum_overlap,
    request_semaphore,
    prompt_debug_path=None,
):
    transcript_path = Path(path)
    rows = read_transcript(transcript_path)
    spans = load_ad_spans(transcript_path, input_root, spans_root, len(rows))
    prepared = prepare_transcript(rows, spans, transcript_path)
    if not prepared:
        return path, []
    text = combine_rows(prepared)

    def messages_for(chunk):
        return messages_for_chunk(chunk, topic_str)

    chunks = transcript_chunks(
        prepared,
        renderer,
        max_model_len,
        max_new_tokens,
        transcript_path,
        messages_for,
    )
    responses = await asyncio.gather(*(
        generate_chunk_responses(
            client,
            model,
            offset,
            chunk,
            max_new_tokens,
            reasoning_effort,
            transcript_path,
            messages_for,
            parse_topics_payload,
            request_semaphore,
            prompt_debug_path,
        )
        for offset, chunk, _rendered in chunks
    ))
    mentions = []
    for _offset, _chunk, payload in (
        result
        for chunk_results in responses
        for result in chunk_results
    ):
        mentions.extend(topic_mentions(payload, transcript_path))
    return path, select_episode_topics(mentions, text, minimum_overlap)


def round_robin_episodes(data_dir):
    show_dirs = [
        d_path for d_path in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d_path)
    ]
    csvs_by_show_dir = {}
    for show_dir in tqdm(show_dirs, desc="Processing shows"):
        csvs_by_show_dir[show_dir] = glob.glob(os.path.join(show_dir, "*.csv"))

    file_names = []
    while csvs_by_show_dir:
        for show in show_dirs:
            if (rem_paths := csvs_by_show_dir.get(show, [])):
                file_names.append(rem_paths.pop())
            else:
                csvs_by_show_dir.pop(show, None)
    return file_names


async def run(
    args, file_names, topic_to_desc, topic_frequencies, topic_str, write_topics, write_quotes,
    processed_files, checkpoint_path,
):
    client = make_openai_client(args)
    request_semaphore = asyncio.Semaphore(REQUEST_CONCURRENCY)
    renderer = make_renderer(args.model, args.max_model_len)
    input_root = Path(args.i).resolve()
    spans_root = Path(args.ad_spans).resolve()
    async def process(path, batch_topic_str):
        try:
            return await process_episode(
                client,
                args.model,
                renderer,
                path,
                batch_topic_str,
                args.max_new_tokens,
                args.reasoning_effort,
                args.max_model_len,
                input_root,
                spans_root,
                args.minimum_overlap,
                request_semaphore,
                args.prompt_debug,
            )
        finally:
            progress.update(1)

    with tqdm(total=len(file_names), desc="Processing texts") as progress:
        for batch in episode_batches(file_names, args.episode_batch):
            topic_str, new_topics, quote_rows = await discover_batch(
                process,
                topic_to_desc,
                topic_str,
                batch,
                topic_frequencies,
                args.max_prompt_topics,
            )
            write_topics(new_topics)
            write_quotes(quote_rows)
            processed_files.extend(
                Path(path).resolve().relative_to(input_root).as_posix()
                for path in batch
            )
            temporary_path = f"{checkpoint_path}.tmp"
            with open(temporary_path, "w", encoding="utf-8") as handle:
                json.dump({"processed_files": processed_files}, handle, indent=2)
                handle.write("\n")
            os.replace(temporary_path, checkpoint_path)


def main(raw_args=None):
    print(f"Process: {os.getpid()}")
    parser = argparse.ArgumentParser(description="Generate candidate topics")
    parser.add_argument("-i", type=os.path.abspath, default="out/whisper_segmented")
    parser.add_argument("-o", type=os.path.abspath)
    parser.add_argument("-n", type=int)
    parser.add_argument(
        "--ad-spans",
        type=Path,
        default=Path("out/ad_spans"),
        help="Directory of ad-span JSON produced by hot_topic.ad_prompt. "
             "Span line numbers must match these transcript CSVs.",
    )
    parser.add_argument("--buffer-size", default=8, type=int)
    parser.add_argument(
        "--max-prompt-topics",
        default=100,
        type=int,
        help="Maximum number of topics to include in each prompt",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume using the processed topics JSON in the output directory",
    )
    parser.add_argument("--minimum-overlap", default=0.95, type=float,
                        help="Minimum quote overlap score from 0 to 1")
    parser.add_argument(
        "--prompt-debug",
        type=Path,
        help="Overwrite this file with the user prompt sent for each LLM request",
    )
    add_vllm_client_arguments(parser)
    args = parser.parse_args(raw_args)

    if not 0 <= args.minimum_overlap <= 1:
        parser.error("--minimum-overlap must be between 0 and 1")
    if args.max_prompt_topics < 1:
        parser.error("--max-prompt-topics must be at least 1")
    args.minimum_overlap *= 100
    validate_vllm_arguments(parser, args)

    input_root = Path(args.i).resolve()
    spans_root = Path(args.ad_spans).resolve()
    if not input_root.is_dir():
        parser.error(f"-i is not a directory: {input_root}")
    if not spans_root.is_dir():
        parser.error(f"--ad-spans is not a directory: {spans_root}")

    file_names = round_robin_episodes(args.i)
    if args.n:
        file_names = file_names[:args.n]

    configure_server(parser, args)
    model_name = args.model
    output_dir = args.o or os.path.abspath(os.path.join("out", f"gen--{normalize_model_name(model_name)}"))
    topic_path = os.path.join(output_dir, "topics.csv")
    quote_path = os.path.join(output_dir, "topic_quotes.csv")
    checkpoint_path = os.path.join(output_dir, CHECKPOINT_FILENAME)
    os.makedirs(output_dir, exist_ok=True)

    if args.resume:
        for output_path in (topic_path, quote_path):
            if not os.path.isfile(output_path):
                parser.error(f"resume output does not exist: {output_path}")
        if not os.path.isfile(checkpoint_path):
            parser.error(f"resume checkpoint does not exist: {checkpoint_path}")
        with open(topic_path, newline="", encoding="utf-8") as handle:
            checkpoint_topics = [
                (row["topic"], row["topic_desc"])
                for row in csv.DictReader(handle)
            ]
        topic_to_desc = OrderedDict(checkpoint_topics)
        try:
            topic_frequencies = quote_topic_frequencies(quote_path)
        except (OSError, csv.Error) as error:
            parser.error(f"could not read quote frequencies from {quote_path}: {error}")
        try:
            with open(checkpoint_path, encoding="utf-8") as handle:
                checkpoint = json.load(handle)
        except json.JSONDecodeError as error:
            parser.error(f"could not parse resume checkpoint {checkpoint_path}: {error}")
        if not isinstance(checkpoint, dict) or "processed_files" not in checkpoint:
            parser.error(
                f"resume checkpoint must contain a processed_files list: {checkpoint_path}"
            )
        processed_files = checkpoint["processed_files"]
        if not all(isinstance(path, str) for path in processed_files):
            parser.error(
                f"resume checkpoint processed_files must be an array of strings: {checkpoint_path}"
            )
        processed_files = list(processed_files)
        processed_file_set = set(processed_files)
        file_names = [
            path
            for path in file_names
            if Path(path).resolve().relative_to(input_root).as_posix()
            not in processed_file_set
        ]
    else:
        topic_to_desc = OrderedDict(DEFAULT_TOPICS)
        topic_frequencies = Counter()
        processed_files = []
    topic_str = topic_prompt(
        topic_to_desc, topic_frequencies, args.max_prompt_topics
    )

    raw_write_topics = get_csv_writer(topic_path, ["author", "topic", "topic_desc"], append=args.resume)
    raw_write_quotes = get_csv_writer(
        quote_path,
        ["author", "episode_file", "topic", "episode_quote"],
        append=args.resume,
    )
    write_topics = lambda rows: raw_write_topics([model_name, *row] for row in rows)
    write_quotes = lambda rows: raw_write_quotes([model_name, *row] for row in rows)

    if not args.resume:
        write_topics([list(topic) for topic in DEFAULT_TOPICS])
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump({"processed_files": processed_files}, handle, indent=2)
            handle.write("\n")

    if file_names:
        asyncio.run(run(
            args,
            file_names,
            topic_to_desc,
            topic_frequencies,
            topic_str,
            write_topics,
            write_quotes,
            processed_files,
            checkpoint_path,
        ))
    else:
        if args.resume:
            with open(checkpoint_path, "w", encoding="utf-8") as handle:
                json.dump({"processed_files": processed_files}, handle, indent=2)
                handle.write("\n")


if __name__ == "__main__":
    main()
