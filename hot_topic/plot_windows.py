#!/usr/bin/env python3

import argparse
import csv
import json
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm


def read_transcript(path):
	with path.open(newline="", encoding="utf-8") as source:
		rows = list(csv.DictReader(source))
	if rows:
		missing = {"start", "end", "text"} - set(rows[0])
		if missing:
			raise ValueError(f"{path} is missing columns: {sorted(missing)}")
	return rows


def _clock_to_seconds(value):
	total = 0.0
	for part in str(value).strip().split(":"):
		total = total * 60 + float(part)
	return total


def timestamps_to_seconds(values, timestamp_unit="auto"):
	texts = [str(value).strip() for value in values]
	if timestamp_unit == "auto" and any(":" in text for text in texts):
		return [_clock_to_seconds(text) for text in texts], "clock"
	if timestamp_unit == "clock":
		return [_clock_to_seconds(text) for text in texts], "clock"

	numbers = [float(text) for text in texts]
	divisors = {"s": 1.0, "cs": 100.0, "ms": 1000.0}
	if timestamp_unit in divisors:
		return [number / divisors[timestamp_unit] for number in numbers], timestamp_unit
	if timestamp_unit != "auto":
		raise ValueError("timestamp_unit must be 'auto', 'clock', 's', 'cs', or 'ms'")

	gaps = [b - a for a, b in zip(numbers, numbers[1:]) if b > a]
	median_gap = float(np.median(gaps)) if gaps else 0.0
	if median_gap >= 500:
		return [number / 1000.0 for number in numbers], "ms"
	if median_gap >= 50:
		return [number / 100.0 for number in numbers], "cs"
	return numbers, "s"


def merge_spans(spans):
	merged = []
	for span in sorted(spans, key=lambda item: (item["start"], item["end"])):
		if merged and span["start"] <= merged[-1]["end"]:
			merged[-1]["end"] = max(merged[-1]["end"], span["end"])
			if span.get("reason") and span["reason"] not in merged[-1].get("reason", ""):
				merged[-1]["reason"] = f"{merged[-1].get('reason', '')}; {span['reason']}".strip("; ")
		else:
			merged.append(dict(span))
	return merged


def load_ad_spans(labels):
	if isinstance(labels, (str, Path)):
		with Path(labels).open(encoding="utf-8") as source:
			labels = json.load(source)
	if isinstance(labels, dict):
		labels = labels.get("spans", [])
	return merge_spans([
		{
			"start": span["start"],
			"end": span["end"],
			"reason": str(span.get("reason", "")),
		}
		for span in labels
	])


def plot_ad_spans(transcript_csv, labels, timestamp_unit="auto"):
	transcript_csv = Path(transcript_csv)
	rows = read_transcript(transcript_csv)
	if not rows:
		figure = go.Figure()
		figure.update_layout(
			title=f"{transcript_csv.name}: advertising spans",
			xaxis_title="Minutes into episode",
			yaxis=dict(visible=False, range=[-1, 1]),
			height=360,
			template="plotly_white",
			showlegend=False,
			hovermode="closest",
		)
		return figure

	starts, unit = timestamps_to_seconds([row["start"] for row in rows], timestamp_unit)
	ends, _ = timestamps_to_seconds([row["end"] for row in rows], timestamp_unit)
	minutes = np.asarray(starts) / 60.0
	spans = load_ad_spans(labels)
	figure = go.Figure()

	for span in spans:
		start, end = span["start"], span["end"]
		if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end < len(rows):
			raise ValueError(f"Span {span!r} is outside transcript line range [0, {len(rows)})")
		hover_text = f"ad span [{start}, {end})"
		if span["reason"]:
			hover_text += f"<br>reason: {span['reason']}"
		span_end = ends[end - 1] / 60.0
		figure.add_trace(
			go.Scatter(
				x=[minutes[start], minutes[start], span_end, span_end, minutes[start]],
				y=[-1, 1, 1, -1, -1],
				fill="toself",
				fillcolor="#e76f51",
				opacity=0.28,
				line=dict(width=0),
				mode="lines",
				hoveron="fills",
				text=hover_text,
				hovertemplate="%{text}<extra></extra>",
				showlegend=False,
			)
		)

	figure.add_trace(
		go.Scatter(
			x=minutes,
			y=np.zeros(len(rows)),
			mode="markers",
			marker=dict(size=3, color="#2f4858"),
			name="Transcript lines",
			text=[row["text"].strip().replace("\n", " ")[:500] for row in rows],
			customdata=np.arange(len(rows)),
			hovertemplate="line %{customdata}<br>%{x:.2f} min<br>%{text}<extra></extra>",
		)
	)
	figure.update_layout(
		title=f"{transcript_csv.name}: advertising spans",
		xaxis_title=f"Minutes into episode ({unit})",
		yaxis=dict(visible=False, range=[-1, 1]),
		height=360,
		template="plotly_white",
		showlegend=False,
		hovermode="closest",
	)
	return figure


def write_ad_plot(task):
	transcript_path, span_path, output_path = task
	figure = plot_ad_spans(transcript_path, span_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	figure.write_html(output_path)
	return output_path


def main(raw_args=None):
	parser = argparse.ArgumentParser(description="Plot advertising spans over transcript timelines")
	parser.add_argument("--data", default=Path("out/whisper_segmented"), type=Path,
						help="Directory containing transcript CSV files")
	parser.add_argument("--spans", "--ad-spans", dest="spans", default=Path("out/ad_spans"), type=Path,
						help="Directory containing span JSON files")
	parser.add_argument("-o", "--output", default=Path("out/ad_plots"), type=Path,
						help="Directory for generated HTML plots")
	parser.add_argument("--skip-finished", action="store_true",
						help="Skip plots whose output HTML file already exists")
	parser.add_argument("--num-workers", type=int, default=max(1, cpu_count() - 1),
						help="Worker processes used to generate plots")
	args = parser.parse_args(raw_args)
	if args.num_workers < 1:
		parser.error("--num-workers must be positive")

	data_root = args.data.resolve()
	spans_root = args.spans.resolve()
	output_root = args.output.resolve()
	if not data_root.is_dir():
		parser.error(f"--data is not a directory: {data_root}")
	if not spans_root.is_dir():
		parser.error(f"--spans is not a directory: {spans_root}")

	tasks = []
	for transcript_path in sorted(data_root.glob("**/*.csv")):
		relative_path = transcript_path.relative_to(data_root)
		span_path = spans_root / relative_path.with_suffix(".json")
		output_path = output_root / relative_path.with_suffix(".html")
		if not span_path.is_file():
			print(f"Warning: missing advertising spans file: {span_path}")
			continue
		if args.skip_finished and output_path.exists():
			continue
		tasks.append((transcript_path, span_path, output_path))

	with Pool(args.num_workers) as pool:
		for output_path in tqdm(
			pool.imap_unordered(write_ad_plot, tasks),
			total=len(tasks),
			desc="Writing advertising plots",
		):
			print(f"Wrote {output_path}")


if __name__ == "__main__":
	main()
