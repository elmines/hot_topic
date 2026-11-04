#!/usr/bin/env python3

import json
import os
from pathlib import Path
import shlex
import stat


JOB_NAME = "silver_labels"
DURATION = "3:00:00"
CPUS = 4
MEMORY = "24gb"


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


def build_script(repo_dir, partition, email):
	commands = "\n".join(
		[
			"uv run python -m hot_topic.silver_label",
			"uv run python -m hot_topic.train_classifier",
			"uv run python -m hot_topic.bert_ad_spans",
		]
	)

	return f"""#!/bin/bash

#SBATCH --time={shell_quote(DURATION)}
#SBATCH --job-name={shell_quote(JOB_NAME)}
#SBATCH --partition={shell_quote(partition)}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={CPUS}
#SBATCH --gres=gpu:1
#SBATCH --mem={shell_quote(MEMORY)}
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

{commands}
"""


def main():
	repo_dir = Path(__file__).resolve().parent
	config = load_config(repo_dir)
	script_path = repo_dir / "slurm_scripts" / f"{JOB_NAME}.sh"
	script_path.parent.mkdir(parents=True, exist_ok=True)

	script = build_script(repo_dir, config["rtx_partition"], config["email"])
	write_code(script_path, script)
	print(f"Wrote script to: {script_path}")


if __name__ == "__main__":
	main()