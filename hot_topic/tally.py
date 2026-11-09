#!/usr/bin/env python3

import argparse
import csv
import os
from collections.abc import Sequence
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

from .utils import ContextExtractor, extract_show_and_episode


MAX_WORKERS = 4
DEFAULT_WORKERS = MAX_WORKERS
_EXTRACT_QUOTE_CONTEXT = None


def _init_worker(input_root, ad_spans, model_names):
    global _EXTRACT_QUOTE_CONTEXT, _MODEL_NAMES
    _MODEL_NAMES = model_names
    _EXTRACT_QUOTE_CONTEXT = (
        ContextExtractor(input_root or Path("out/whisper_segmented"), ad_spans)
        if ad_spans is not None
        else ContextExtractor()
    )


def _process_quote(task):
    quote_row, row_decisions, row_explanations = task
    show_id, episode_id = extract_show_and_episode(quote_row["episode_file"])
    row = [
        show_id,
        episode_id,
        quote_row["topic"],
        _EXTRACT_QUOTE_CONTEXT(quote_row),
        quote_row['episode_quote'],
        quote_row["author"],
    ]
    row.extend(row_decisions)
    row.append(combine_decisions(row_decisions, _MODEL_NAMES))
    row.extend(row_explanations)
    return row


_MODEL_NAMES = ()


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("/", "--")


def read_topic_quotes(paths: Sequence[str]) -> list[dict[str, str]]:
    rows = []
    for path in paths:
        with open(path, newline="") as source:
            reader = csv.DictReader(source)
            required = {"author", "episode_file", "topic", "episode_quote"}
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f"{path} is missing one or more topic quote columns")
            rows.extend(reader)
    return rows


def read_rejections(paths: Sequence[str]) -> tuple[list[str], list[list[str]]]:
    model_names = []
    decisions_by_model = []
    for path in paths:
        with open(path, newline="") as source:
            reader = csv.reader(source)
            header = next(reader, None)
            if not header or len(header) != 2:
                raise ValueError(f"{path} must have a two-column rejection header")
            rejected_header, explanation_header = header
            suffix = "--rejected"
            if not rejected_header.endswith(suffix):
                raise ValueError(f"{path} has an invalid rejection column name")
            model_name = rejected_header[:-len(suffix)]
            expected_explanation = f"{model_name}--explanation"
            if explanation_header != expected_explanation:
                raise ValueError(f"{path} has mismatched rejection columns")
            if model_name in model_names:
                raise ValueError(f"Duplicate rejection model: {model_name}")
            model_names.append(model_name)
            decisions_by_model.append(list(reader))
    return model_names, decisions_by_model


def parse_decision(value: str, path: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False
    raise ValueError(f"Invalid rejection decision in {path}, row {row_number}: {value!r}")


def combine_decisions(decisions: list[bool], model_names: Sequence[str]) -> bool:
    true_count = sum(decisions)
    false_count = len(decisions) - true_count
    if true_count != false_count:
        return true_count > false_count
    for decision, model_name in zip(decisions, model_names):
        if "gpt-oss" in model_name.lower():
            return decision
    return decisions[0]


def tally(
    topic_quote_paths: Sequence[str],
    rejection_paths: Sequence[str],
    ad_spans: str | os.PathLike[str] | None = None,
    input_root: str | os.PathLike[str] | None = None,
    worker_count: int = DEFAULT_WORKERS,
) -> tuple[list[str], list[list[object]]]:
    quote_rows = read_topic_quotes(topic_quote_paths)
    model_names, raw_rejections = read_rejections(rejection_paths)
    if not model_names:
        raise ValueError("At least one rejection CSV is required")
    if any(len(rows) != len(quote_rows) for rows in raw_rejections):
        raise ValueError("Each rejection CSV must contain one row per topic quote")

    decisions = []
    explanations = []
    for rows, path in zip(raw_rejections, rejection_paths):
        model_decisions = []
        model_explanations = []
        for row_number, row in enumerate(rows, start=2):
            if len(row) != 2:
                raise ValueError(f"{path}, row {row_number} must have two columns")
            model_decisions.append(parse_decision(row[0], path, row_number))
            model_explanations.append(row[1])
        decisions.append(model_decisions)
        explanations.append(model_explanations)

    header = ["show_id", "episode_id", "topic", "contextualized_quote", "episode_quote", "author"]
    header.extend(f"{name}--rejected" for name in model_names)
    header.append("combined--rejected")
    header.extend(f"{name}--explanation" for name in model_names)

    if not 1 <= worker_count <= MAX_WORKERS:
        raise ValueError(f"worker_count must be between 1 and {MAX_WORKERS}")

    global _MODEL_NAMES
    _MODEL_NAMES = model_names
    def tasks():
        for index, quote_row in enumerate(quote_rows):
            yield (
                quote_row,
                [model_rows[index] for model_rows in decisions],
                [model_rows[index] for model_rows in explanations],
            )

    with Pool(
        worker_count,
        initializer=_init_worker,
        initargs=(input_root, ad_spans, model_names),
    ) as pool:
        output_rows = list(tqdm(
            pool.imap(_process_quote, tasks(), chunksize=32),
            total=len(quote_rows),
            desc="Processing quotes",
        ))
    return header, output_rows


def main(raw_args=None):
    parser = argparse.ArgumentParser(description="Combine topic quote rejection decisions")
    parser.add_argument("-i", "--input", "--quotes", dest="quotes", nargs="+", required=True,
                        type=os.path.abspath, help="topic_quotes.csv files generated by gen.py")
    parser.add_argument("-r", "--reject", "--rejections", dest="rejections", nargs="+", required=True,
                        type=os.path.abspath, help="CSV files generated by reject_quote.py")
    parser.add_argument("--root", default=Path("out/whisper_segmented"), type=Path,
                        help="Root directory for transcript paths in the input CSVs")
    parser.add_argument("--ad-spans", type=Path,
                        help="Directory of ad-span JSON produced by hot_topic.ad_prompt")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_WORKERS,
                        help=f"Worker processes used to process quotes (1-{MAX_WORKERS})")
    parser.add_argument("-o", "--output", required=True, type=os.path.abspath)
    args = parser.parse_args(raw_args)

    if args.ad_spans is not None:
        root = args.root.resolve()
        if not root.is_dir():
            parser.error(f"--root is not a directory: {root}")
        ad_spans = args.ad_spans.resolve()
        if not ad_spans.is_dir():
            parser.error(f"--ad-spans is not a directory: {ad_spans}")
    else:
        root = None
        ad_spans = None

    if not 1 <= args.num_workers <= MAX_WORKERS:
        parser.error(f"--num-workers must be between 1 and {MAX_WORKERS}")

    header, rows = tally(args.quotes, args.rejections, ad_spans, root, args.num_workers)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(header)
        writer.writerows(rows)


if __name__ == "__main__":
    main()