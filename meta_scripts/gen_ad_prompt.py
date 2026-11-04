#!/usr/bin/env python3

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import stat


# Recent N-worker set of jobs
# took 5 hours a piece for 1615+ transcripts each
# FILES_PER_HOUR = 1615 / 5.5
FILES_PER_HOUR = 1615 / 7


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
        description="Generate balanced Slurm scripts for ad_prompt jobs."
    )
    parser.add_argument("--root", default=Path("out/whisper_segmented"), type=Path,
                        help="Root directory containing transcript CSV files")
    parser.add_argument("--out", default=Path("out/ad_spans"), type=Path,
                        help="Directory for ad span JSON outputs")
    parser.add_argument("-n", type=int, required=True,
                        help="Number of Slurm jobs to generate")
    return parser.parse_args()


def load_config(repo_dir):
    with (repo_dir / "config.json").open() as handle:
        return json.load(handle)


def discover_csv_files(root):
    return sorted(path for path in root.rglob("*.csv") if path.is_file())


def balance_files(files, num_jobs):
    buckets = [{"files": [], "bytes": 0} for _ in range(num_jobs)]
    for path in sorted(files, key=lambda item: (-item.stat().st_size, str(item))):
        bucket = min(buckets, key=lambda item: (item["bytes"], len(item["files"])))
        bucket["files"].append(path)
        bucket["bytes"] += path.stat().st_size
    return buckets


def format_duration(hours):
    total_seconds = math.ceil(hours * 60 * 60)
    hours, remainder = divmod(total_seconds, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}"


def build_script(repo_dir, root, output, duration, partition, email, input_paths, job_number, port_number):
    if input_paths:
        command = [
            "uv",
            "run",
            "python",
            "-m",
            "hot_topic.ad_prompt",
            "--base-url",
            f"http://0.0.0.0:{port_number}/v1",
            "--root",
            shell_quote(root),
            "--output",
            shell_quote(output),
            "-i",
            *(shell_quote(path) for path in input_paths),
        ]
        command_str = " \\\n\t".join(command)
    else:
        command_str = "echo 'No files assigned to this job.'"

    return f"""#!/bin/bash

#SBATCH --time={shell_quote(duration)}
#SBATCH --job-name=ad_prompt_{job_number:03d}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --mail-user={shell_quote(email)}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load cuda/13.0.2 git
export XDG_RUNTIME_DIR=$SLURM_TMPDIR
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
date
hostname
script_dir=$(realpath $(dirname $0))
cd {shell_quote(repo_dir)}
pwd

server_pid=""
cleanup() {{
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}}
trap cleanup EXIT

vllm_log_path={shell_quote(repo_dir)}/slurm_scripts/$SLURM_JOB_NAME.$SLURM_JOB_ID.vllm.log
RUN_PREFIX="exec" API_PORT={port_number} server_scripts/gpt.sh >$vllm_log_path 2>&1 &
server_pid=$!
echo Writing vllm output to $vllm_log_path

{command_str}
"""


def main():
    args = parse_args()
    if args.n <= 0:
        raise SystemExit("-n must be a positive integer")

    repo_dir = Path(__file__).resolve().parent
    root = args.root.expanduser().resolve()
    output = args.out.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist or is not a directory: {root}")
    if output == root or output.is_relative_to(root) or root.is_relative_to(output):
        raise SystemExit("--out must be a different directory from --root")

    config = load_config(repo_dir)
    csv_files = discover_csv_files(root)
    buckets = balance_files(csv_files, args.n)
    max_files = max((len(bucket["files"]) for bucket in buckets), default=0)
    duration = format_duration(max(1, max_files) / FILES_PER_HOUR)
    slurm_dir = repo_dir / "slurm_scripts"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    for job_number, bucket in enumerate(buckets, start=1):
        port_number = 21699 + job_number
        script_path = slurm_dir / f"ad_prompt_{job_number:03d}_of_{args.n:03d}.sh"
        script = build_script(
            repo_dir,
            root,
            output,
            duration,
            config["rtx_partition"],
            config["email"],
            bucket["files"],
            job_number,
            port_number
        )
        write_code(script_path, script)
        print(f"Job {job_number:03d}: {len(bucket['files'])} files, {bucket['bytes']} bytes")
        print(f"Wrote script to: {script_path}")


if __name__ == "__main__":
    main()
