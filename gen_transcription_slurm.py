#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys


DEFAULT_TIME_LIMIT = "3:00:00"
DEFAULT_REAL_TIME_FACTOR = 30.0


def chmodx(out_path):
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def write_code(out_path, bash_code):
    with open(out_path, "w") as handle:
        handle.write(bash_code)
    chmodx(out_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate balanced Slurm scripts for whisper-cli transcription jobs."
    )
    parser.add_argument("num_jobs", type=int, help="Number of Slurm jobs to generate")
    parser.add_argument(
        "input_dir",
        help="Root directory containing nested MP3/M4A files to transcribe",
    )
    parser.add_argument(
        "--real-time-factor",
        type=float,
        default=DEFAULT_REAL_TIME_FACTOR,
        help=(
            "Estimated transcription speed as audio-seconds per real second "
            f"(default: {DEFAULT_REAL_TIME_FACTOR})"
        ),
    )
    return parser.parse_args()


def load_config(repo_dir):
    with (repo_dir / "config.json").open() as handle:
        config = json.load(handle)
    return config["email"], config["l4_partition"], config["whisper_root"]


def discover_audio_files(input_dir):
    discovered = []
    for path in input_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".mp3", ".m4a"}:
            if path.name.startswith("."):
                # print(f"Warning: ignoring hidden audio file: {path}", file=sys.stderr)
                continue
            discovered.append(path)
    return sorted(discovered)


def output_csv_path(out_dir, input_dir, audio_path):
    try:
        relative_path = audio_path.relative_to(input_dir)
    except Exception:
        # Fallback to a safe relative path if audio_path is not a subpath
        relative_path = Path(os.path.relpath(audio_path, input_dir))

    # Build the output path preserving the input directory structure
    return out_dir / relative_path.parent / f"{relative_path.name}.csv"


def balance_files(files, num_jobs):
    buckets = [{"files": [], "bytes": 0} for _ in range(num_jobs)]
    for audio_path, size_bytes, output_path, is_m4a_like in sorted(files, key=lambda item: (-item[1], str(item[0]))):
        bucket = min(buckets, key=lambda item: (item["bytes"], len(item["files"])))
        bucket["files"].append((audio_path, size_bytes, output_path, is_m4a_like))
        bucket["bytes"] += size_bytes
    return buckets


def shell_quote(value):
    return shlex.quote(str(value))


def parse_duration_to_seconds(duration_text):
    match = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", duration_text)
    if not match:
        return None
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def classify_audio_codec(codec_name, stderr_text):
    codec_lower = (codec_name or "").lower()
    stderr_lower = stderr_text.lower()

    if codec_lower.startswith("mp3"):
        return False
    if (
        codec_lower.startswith("aac")
        or codec_lower.startswith("alac")
        or "mp4a" in codec_lower
        or "m4a" in stderr_lower
    ):
        return True
    return None


def probe_audio_file(audio_path, ffmpeg_bin):
    try:
        result = subprocess.run(
            [ffmpeg_bin, "-i", str(audio_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        return None

    stderr_text = result.stderr.decode("utf-8", errors="replace")
    duration_match = re.search(r"Duration:\s*(\d+:\d+:\d+(?:\.\d+)?)", stderr_text)
    if not duration_match:
        return None, None

    duration_seconds = parse_duration_to_seconds(duration_match.group(1))
    if duration_seconds is None:
        return None, None

    codec_match = re.search(r"Audio:\s*([^,\s]+)", stderr_text)
    if not codec_match:
        return duration_seconds, None

    is_m4a_like = classify_audio_codec(codec_match.group(1), stderr_text)
    return duration_seconds, is_m4a_like


def format_slurm_time(seconds):
    total_seconds = max(60, int(math.ceil(seconds)))
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    return f"{hours}:{minutes:02d}:{secs:02d}"


def estimate_job_time_limit(job_files, duration_by_path, real_time_factor):
    audio_seconds = 0.0
    for audio_path, _size_bytes, _output_path, is_m4a_like in job_files:
        duration_seconds = duration_by_path.get(audio_path, 0.0)
        if is_m4a_like:
            duration_seconds *= 1.10
        audio_seconds += duration_seconds

    estimated_runtime_seconds = (audio_seconds / real_time_factor) * 1.10
    return format_slurm_time(estimated_runtime_seconds)


def build_command_block(job_files, repo_dir, whisper_root):
    if not job_files:
        return "echo 'No files assigned to this job.'"

    lines = [
        f"WHISPER_ROOT={shell_quote(whisper_root)}",
        f"REPO_DIR={shell_quote(repo_dir)}",
        'FFMPEG_BIN="ffmpeg"',
        'WHISPER_BIN="$WHISPER_ROOT/build/bin/whisper-cli"',
        'WHISPER_MODEL="$WHISPER_ROOT/models/ggml-medium.bin"',
        'cd "$REPO_DIR"',
        'source_files=(',
    ]

    for audio_path, _size_bytes, _output_path, _is_m4a_like in job_files:
        lines.append(f"  {shell_quote(audio_path)}")

    lines.extend(
        [
            ")",
            "output_files=(",
        ]
    )

    for _audio_path, _size_bytes, output_path, _is_m4a_like in job_files:
        lines.append(f"  {shell_quote(output_path)}")

    lines.extend(
        [
            ")",
            "m4a_files=(",
        ]
    )

    for audio_path, _size_bytes, _output_path, is_m4a_like in job_files:
        if is_m4a_like:
            lines.append(f"  {shell_quote(audio_path)}")

    lines.extend(
        [
            ")",
            'declare -A m4a_map=()',
            'for m4a_source in "${m4a_files[@]}"; do',
            '  m4a_map["$m4a_source"]=1',
            'done',
            'for index in "${!source_files[@]}"; do',
            '  source="${source_files[$index]}"',
            '  destination="${output_files[$index]}"',
            '  mkdir -p "$(dirname \"$destination\")"',
            '  if [ -f "$destination" ]; then',
            '    echo "Skipping existing $destination"',
            '    continue',
            '  fi',
            '  transcribe_input="$source"',
            '  temp_wav=""',
            '  if [[ -n "${m4a_map[$source]+x}" ]]; then',
            '    temp_wav="$SLURM_TMPDIR/whisper_${SLURM_JOB_ID}_${index}_$(basename "$source").wav"',
            '    echo "Converting M4A-like input $source -> $temp_wav"',
            '    "$FFMPEG_BIN" -y -i "$source" -acodec pcm_s16le -ar 16000 -ac 1 "$temp_wav" > /dev/null 2>&1',
            '    if [ ! -f "$temp_wav" ]; then',
            '      echo "Failed to create wav for $source"',
            '      continue',
            '    fi',
            '    transcribe_input="$temp_wav"',
            '  fi',
            '  source_dir="$(dirname "$transcribe_input")"',
            '  csv_name="$(basename "$transcribe_input").csv"',
            '  source_csv="$source_dir/$csv_name"',
            '  echo "Transcribing $source"',
            '  rm -f "$source_csv"',
            '  "$WHISPER_BIN" --beam-size 1 -ocsv -np --model "$WHISPER_MODEL" "$transcribe_input" > /dev/null',
            '  if [ -f "$source_csv" ]; then',
            '    mv "$source_csv" "$destination"',
            '  else',
            '    echo "Missing csv output for $source"',
            '  fi',
            '  if [ -n "$temp_wav" ]; then',
            '    rm -f "$temp_wav"',
            '  fi',
            'done',
        ]
    )

    return "\n".join(lines)


slurm_template = """#!/bin/bash

#SBATCH --time={time_limit}
#SBATCH --job-name={name}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16gb
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export XDG_RUNTIME_DIR=$SLURM_TMPDIR
echo CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
module load cuda/12.4.1 gcc/12.2.0 ffmpeg
date
hostname
cd {repo_dir}
pwd

{command}
"""


def main():
    args = parse_args()
    if args.num_jobs <= 0:
        raise SystemExit("num_jobs must be a positive integer")
    if args.real_time_factor <= 0:
        raise SystemExit("real-time-factor must be a positive number")

    repo_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"input_dir does not exist or is not a directory: {input_dir}")

    out_dir = repo_dir / "out" / "whisper_segmented"
    slurm_dir = repo_dir / "slurm_scripts"
    out_dir.mkdir(parents=True, exist_ok=True)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    for existing_script in slurm_dir.glob("*.sh"):
        existing_script.unlink()

    user_email, l4_part, whisper_root = load_config(repo_dir)

    discovered = discover_audio_files(input_dir)
    pending = []
    skipped_existing = 0
    for audio_path in discovered:
        output_path = output_csv_path(out_dir, input_dir, audio_path)
        if output_path.exists():
            skipped_existing += 1
            continue
        pending.append((audio_path, audio_path.stat().st_size, output_path))

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin is None:
        raise SystemExit("ffmpeg not found in PATH; ffmpeg is required")

    duration_by_path = {}
    dropped_unparseable = 0
    dropped_unsupported_codec = 0
    filtered_pending = []
    for audio_path, _size_bytes, _output_path in pending:
        duration_seconds, is_m4a_like = probe_audio_file(audio_path, ffmpeg_bin)
        if duration_seconds is None:
            dropped_unparseable += 1
            print(
                f"Warning: could not parse duration for {audio_path}; assuming file is corrupted and skipping.",
                file=sys.stderr,
            )
            continue
        if is_m4a_like is None:
            dropped_unsupported_codec += 1
            print(
                f"Warning: unsupported codec for {audio_path}; expected MP3 or M4A-like audio and skipping.",
                file=sys.stderr,
            )
            continue
        duration_by_path[audio_path] = duration_seconds
        filtered_pending.append((audio_path, _size_bytes, _output_path, is_m4a_like))
    pending = filtered_pending

    buckets = balance_files(pending, args.num_jobs)
    name_width = max(2, len(str(args.num_jobs)))

    for index, bucket in enumerate(buckets, start=1):
        job_name = f"whisper_{index:0{name_width}d}_of_{args.num_jobs:0{name_width}d}"
        script_path = slurm_dir / f"{job_name}.sh"
        command = build_command_block(bucket["files"], repo_dir, whisper_root)
        time_limit = estimate_job_time_limit(bucket["files"], duration_by_path, args.real_time_factor)
        script = slurm_template.format(
            time_limit=time_limit,
            name=job_name,
            partition=l4_part,
            user_email=user_email,
            repo_dir=repo_dir,
            command=command,
        )
        write_code(script_path, script)

    print(f"Discovered audio files (.mp3/.m4a): {len(discovered)}")
    if skipped_existing:
        print(f"Skipped existing outputs: {skipped_existing}")
    if dropped_unparseable:
        print(f"Dropped as corrupted: {dropped_unparseable}")
    if dropped_unsupported_codec:
        print(f"Dropped unsupported codec: {dropped_unsupported_codec}")
    print(f"Queued for generation: {len(pending)}")
    print(
        "Used ffmpeg durations with real-time-factor="
        f"{args.real_time_factor}, 10% M4A conversion penalty, and 10% runtime margin."
    )
    for index, bucket in enumerate(buckets, start=1):
        print(f"Job {index:0{name_width}d}: {len(bucket['files'])} files, {bucket['bytes']} bytes")
    print(f"Wrote scripts to: {slurm_dir}")


if __name__ == "__main__":
    main()
