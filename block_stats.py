#!/usr/bin/env python3
import csv
import pdb
import sys
import glob
from pathlib import Path
import re
import subprocess

from tqdm import tqdm

DECIMAL_FORMAT = "{:.3f}"

BAD_LEN_SENTINEL = "filesize and duration do not match"

def probe_audio_file(audio_path):
    result = subprocess.run(
        ["ffmpeg", "-i", str(audio_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    stderr_text = result.stderr.decode("utf-8", errors="replace")
    duration_match = re.search(r"Duration:\s*(\d+:\d+:\d+(?:\.\d+)?)", stderr_text)
    duration_text = duration_match.group(1)
    match = re.match(r"^(\d+):(\d+):(\d+(?:\.\d+)?)$", duration_text)
    hours = int(match.group(1))
    minutes = int(match.group(2))
    seconds = float(match.group(3))
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if BAD_LEN_SENTINEL in stderr_text:
        print(f"{audio_path} has incorrect duration metadata", file=sys.stderr)
        return -1
    # Want millis
    return total_seconds * 1000

MUSIC_PATT = re.compile(
    r"♪|"
    r"[\[(](music|foreign|upbeat|singing|rock|sirens wailing|speaking|applause|blank[ _]audio)|"
    r'\[ silence \]|'
    r"(music|foreign language)[\]\)]",
    re.IGNORECASE
)

def trans_stats(transcript_path, audio_path, repeat_thresh = 15000):
    with open(transcript_path, 'r') as r:
        rows = list(csv.DictReader(r))
    rows = [
        {
            'start': int(r['start']),
            'end': int(r['end']),
            'text': r['text']
        }
        for r in rows
    ]
    rdict = {
        "transcript": transcript_path,
        "valid_audio": False,
    }
    trans_size = probe_audio_file(audio_path)
    if trans_size < 0:
        return rdict

    appended_millis = max(0, rows[-1]['end'] - trans_size)
    replaced_millis = 0
    music_millis = 0

    seg_timestamp = 0
    seg_index = 0

    for i, row in enumerate(rows[1:], start=1):
        if row['start'] >= trans_size:
            break
        row['end'] = min(row['end'], trans_size)
        if row['text'] != (prev_text := rows[i-1]['text']):
            # Index check accounts for a single line being longer in duration than repeat_thresh
            # Music check ignores musical data (assume those are not transcription errors)
            if i > seg_index + 1 and (seg_dur := row['start'] - seg_timestamp) >= repeat_thresh:
                if MUSIC_PATT.search(prev_text):
                    music_millis += seg_dur
                else:
                    replaced_millis += seg_dur
            seg_timestamp = row['start']
            seg_index = i
    rdict.update({
        "valid_audio": True,
        "percent_replaced_text": DECIMAL_FORMAT.format(replaced_millis / trans_size),
        "percent_replaced_music": DECIMAL_FORMAT.format(music_millis / trans_size),
        "percent_added": DECIMAL_FORMAT.format(appended_millis / trans_size)
    })
    return rdict


if __name__ == "__main__":
    trans_root = Path("out/whisper_segmented")
    audio_root = Path("/blue/bonniejdorr/ethanlmines/PodcastDataset/audio/")
    transcript_paths = glob.glob( str(trans_root / "*/*.csv") )

    writer = csv.DictWriter(sys.stdout, fieldnames=["valid_audio", "percent_replaced_text", "percent_replaced_music", "percent_added", "transcript"])
    writer.writeheader()
    # transcript_paths = ["out/whisper_segmented/711095/711095_00008.mp3.csv"]
    for trans_path in map(Path, tqdm(transcript_paths)):
        audio_path = str(audio_root / trans_path.relative_to(trans_root)).replace('.csv', '')
        writer.writerow(trans_stats(trans_path, audio_path))
