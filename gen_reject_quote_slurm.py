#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
import shlex
import stat


JOBS = [
    ("8:00:00", "meta-llama/Llama-3.3-70B-Instruct", "llama.sh", "b200", "gpu:1"),
    ("4:00:00", "openai/gpt-oss-120b", "gpt.sh", "rtx", "gpu:1"),
    ("4:00:00", "google/gemma-4-31B-it", "gemma.sh", "rtx", "gpu:1"),
]


def shell_quote(value):
    return shlex.quote(str(value))


def chmodx(out_path):
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_code(out_path, bash_code):
    with out_path.open("w") as handle:
        handle.write(bash_code)
    chmodx(out_path)


def load_config(repo_dir):
    with (repo_dir / "config.json").open() as handle:
        return json.load(handle)


def build_script(repo_dir, root, duration, partition, email, input_paths, port_number, model, server_script, gres):
    if input_paths:
        command = [
            "uv", "run", "python", "-m",             "hot_topic.reject_quote",
            "--base-url", f"http://0.0.0.0:{port_number}/v1",
            "--root", shell_quote(root),
            "-i", *(shell_quote(path) for path in input_paths),
        ]
        command_str = " \\\n\t".join(command)
    else:
        command_str = "echo 'No files assigned to this job.'"

    return f"""#!/bin/bash

#SBATCH --time={shell_quote(duration)}
#SBATCH --job-name=reject_quote_{shell_quote(model.replace('/', '--'))}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres={shell_quote(gres)}
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
MODEL={shell_quote(model)} RUN_PREFIX="exec" API_PORT={port_number} server_scripts/{shell_quote(server_script)} >$vllm_log_path 2>&1 &
server_pid=$!
echo Writing vllm output to $vllm_log_path

{command_str}
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Slurm scripts for reject_quote jobs.")
    parser.add_argument("--root", default=Path("out"), type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    repo_dir = Path(__file__).resolve().parent
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"root does not exist or is not a directory: {root}")
    config = load_config(repo_dir)
    slurm_dir = repo_dir / "slurm_scripts"
    slurm_dir.mkdir(parents=True, exist_ok=True)

    partitions = {"rtx": config["rtx_partition"], "b200": config["b200_partition"]}
    input_paths = [
        root / f"gen--{model.replace('/', '--')}" / "topic_quotes.csv"
        for _, model, _, _, _ in JOBS
    ]
    for job_number, (duration, model, server_script, partition_name, gres) in enumerate(JOBS, start=1):
        port_number = 23200 + job_number
        script_path = slurm_dir / f"reject_quote_{model.replace('/', '--')}.sh"
        script = build_script(
            repo_dir, root, duration, partitions[partition_name],
            config["email"], input_paths, port_number,
            model, server_script, gres,
        )
        write_code(script_path, script)
        print(f"Model: {model}")
        print(f"Wrote script to: {script_path}")


if __name__ == "__main__":
    main()
