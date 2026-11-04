#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shlex
import stat


GPU_HOUR_BUDGET = 80


def shell_quote(value):
    return shlex.quote(str(value))


def chmodx(out_path):
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_code(out_path, bash_code):
    with out_path.open("w") as handle:
        handle.write(bash_code)
    chmodx(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate balanced Slurm scripts for line_score jobs."
    )
    parser.add_argument("num_jobs", type=int, help="Number of Slurm jobs to generate")
    parser.add_argument("--root", required=True, type=Path, help="Root directory containing CSV files")
    parser.add_argument("-t", "--topics", required=True, type=Path, help="Topics CSV passed to line_score")
    return parser.parse_args()


def load_config(repo_dir):
    with (repo_dir / "config.json").open() as handle:
        return json.load(handle)


def discover_csv_files(root):
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def count_rows(path):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    return len(rows)


def balance_files(files, num_jobs):
    buckets = [{"files": [], "rows": 0} for _ in range(num_jobs)]
    for path, row_count in sorted(files, key=lambda item: (-item[1], str(item[0]))):
        bucket = min(buckets, key=lambda item: (item["rows"], len(item["files"])))
        bucket["files"].append((path, row_count))
        bucket["rows"] += row_count
    return buckets


def format_duration(hours):
    total_seconds = math.ceil(hours * 60 * 60)
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def build_script(repo_dir, root, topics, duration, partition, email, input_paths, job_number):
    if input_paths:
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "hot_topic.line_score",
            "--root",
            shell_quote(root),
            "-t",
            shell_quote(topics),
            "--files",
            *(shell_quote(path) for path in input_paths),
        ]
        command_str = " \\\n\t".join(command)
    else:
        command_str = "echo 'No files assigned to this job.'"

    return f"""#!/bin/bash

#SBATCH --time={shell_quote(duration)}
#SBATCH --job-name=line_score_{job_number:03d}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=24gb
#SBATCH --mail-user={shell_quote(email)}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load cuda/13.0.2 git
export XDG_RUNTIME_DIR=$SLURM_TMPDIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
date
hostname
cd {shell_quote(repo_dir)}
pwd

{command_str}
"""


def main():
    args = parse_args()
    if args.num_jobs <= 0:
        raise SystemExit("num_jobs must be a positive integer")

    repo_dir = Path(__file__).resolve().parent
    root = args.root.expanduser().resolve()
    topics = args.topics.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist or is not a directory: {root}")
    if not topics.is_file():
        raise SystemExit(f"topics file does not exist: {topics}")

    config = load_config(repo_dir)
    csv_files = [(path, count_rows(path)) for path in discover_csv_files(root)]
    buckets = balance_files(csv_files, args.num_jobs)
    duration = format_duration(GPU_HOUR_BUDGET / args.num_jobs)
    slurm_dir = repo_dir / "slurm_scripts"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    for job_number, bucket in enumerate(buckets, start=1):
        script_path = slurm_dir / f"line_score_{job_number:03d}_of_{args.num_jobs:03d}.sh"
        script = build_script(
            repo_dir,
            root,
            topics,
            duration,
            config["l4_partition"],
            config["email"],
            [path for path, _row_count in bucket["files"]],
            job_number,
        )
        write_code(script_path, script)
        print(f"Job {job_number:03d}: {len(bucket['files'])} files, {bucket['rows']} rows")
        print(f"Wrote script to: {script_path}")


if __name__ == "__main__":
    main()
