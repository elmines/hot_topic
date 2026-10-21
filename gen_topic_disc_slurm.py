#!/usr/bin/env python3

import json
import os
from pathlib import Path
import shlex
import stat


JOBS = [
	("18:00:00", "meta-llama/Llama-3.3-70B-Instruct", "b200", "gpu:1"),
	("15:00:00", "openai/gpt-oss-120b"              , "b200", "gpu:1"),
	("15:00:00", "google/gemma-4-31B-it"            , "b200", "gpu:1"),
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


def build_script(repo_dir, duration, partition, email, model, gres, port):
	commands = [
		" \\\n\t".join(
		[
			"uv",
			"run",
			"python",
			"-m",
			"hot_topic.gen",
			"-i",
			shell_quote(repo_dir / "out" / "resegmented"),
			"--ad-spans",
			shell_quote(repo_dir / "out" / "ad_spans"),
			"--port",
			str(port),
		]
		),
	]
	command_str = "\n".join(commands)

	return f"""#!/bin/bash

#SBATCH --time={shell_quote(duration)}
#SBATCH --job-name=topic_disc_{shell_quote(model.replace('/', '--'))}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres={shell_quote(gres)}
#SBATCH --mem=48gb
#SBATCH --mail-user={shell_quote(email)}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

module load cuda/13.0.2 git
export XDG_RUNTIME_DIR=$SLURM_TMPDIR
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
date
hostname
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
MODEL={shell_quote(model)} RUN_PREFIX="exec" API_PORT={port} server_scripts/gpt.sh >$vllm_log_path 2>&1 &
server_pid=$!
echo Writing vllm output to $vllm_log_path

{command_str}
"""


def main():
	repo_dir = Path(__file__).resolve().parent
	config = load_config(repo_dir)
	partitions = {
		"rtx": config["rtx_partition"],
		"b200": config["b200_partition"],
	}
	for job_number, (duration, model, partition_name, gres) in enumerate(JOBS, start=1):
		model_dir_name = model.replace("/", "--")
		script_path = repo_dir / "slurm_scripts" / f"topic_disc_{model_dir_name}.sh"
		script_path.parent.mkdir(parents=True, exist_ok=True)

		script = build_script(
			repo_dir,
			duration,
			partitions[partition_name],
			config["email"],
			model,
			gres,
			23100 + job_number,
		)
		write_code(script_path, script)
		print(f"Wrote script to: {script_path}")


if __name__ == "__main__":
	main()
