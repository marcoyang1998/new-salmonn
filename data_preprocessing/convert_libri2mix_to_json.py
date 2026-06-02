"""
Convert Libri2Mix metadata CSV files to training JSON format with overlap/diarization prompts.
Time tokens range: 0.00 to 60.00 with granularity 0.10.
"""

import csv
import json
import random
import os

PROMPTS_FILE = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN_wenyi/overlap_prompts.txt"
OUTPUT_DIR = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN_wenyi/salmonn_data_v1.1/librimix_timestamps"

# Each entry: (base_dir, csv_filename, output_json_filename)
SPLITS = [
    # Libri2Mix
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri2Mix_delay_3.0/wav16k/max/metadata", "transcript_test.csv", "libri2mix_overlap_test.json"),
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri2Mix_delay_3.0/wav16k/max/metadata", "transcript_dev.csv", "libri2mix_overlap_dev.json"),
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri2Mix_delay_3.0/wav16k/max/metadata", "transcript_train-100.csv", "libri2mix_overlap_train100.json"),
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri2Mix_delay_3.0/wav16k/max/metadata", "transcript_train-360.csv", "libri2mix_overlap_train360.json"),
    # Libri3Mix
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri3Mix_delay_4.0/wav16k/max/metadata", "transcript_test.csv", "libri3mix_overlap_test.json"),
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri3Mix_delay_4.0/wav16k/max/metadata", "transcript_dev.csv", "libri3mix_overlap_dev.json"),
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri3Mix_delay_4.0/wav16k/max/metadata", "transcript_train-100.csv", "libri3mix_overlap_train100.json"),
    ("/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri3Mix_delay_4.0/wav16k/max/metadata", "transcript_train-360.csv", "libri3mix_overlap_train360.json"),
]


def round_to_token(t: float) -> float:
    """Round time to nearest 0.10 and clamp to [0.00, 60.00]."""
    t = max(0.0, min(60.0, t))
    return round(round(t / 0.1) * 0.1, 2)


def format_time_token(t: float) -> str:
    return f"<|{t:.2f}|>"


def format_speaker_token(speaker_idx: int) -> str:
    return f"<|speaker{speaker_idx}|>"


def build_output(segments):
    """Build the diarization output string sorted by start time.

    Args:
        segments: list of (start, end, transcript) tuples.
    """
    # Sort by start time; assign speaker labels in chronological order
    sorted_segs = sorted(segments, key=lambda x: x[0])

    parts = []
    for idx, (start, end, transcript) in enumerate(sorted_segs, start=1):
        start_tok = format_time_token(round_to_token(start))
        end_tok = format_time_token(round_to_token(end))
        speaker_tok = format_speaker_token(idx)
        parts.append(f"{start_tok}{speaker_tok}{transcript}{end_tok}")

    return "".join(parts)


def detect_num_speakers(fieldnames: list) -> int:
    """Detect number of speakers from CSV header."""
    n = 1
    while f"source_{n + 1}_transcript" in fieldnames:
        n += 1
    return n


def convert_csv(csv_path: str, prompts: list) -> list:
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        num_speakers = detect_num_speakers(reader.fieldnames)
        for row in reader:
            audio_path = row["mixture_path"]

            segments = []
            for i in range(1, num_speakers + 1):
                transcript = row[f"source_{i}_transcript"].strip().capitalize()
                start = float(row[f"source_{i}_start_time"])
                end = float(row[f"source_{i}_end_time"])
                segments.append((start, end, transcript))

            prompt = random.choice(prompts)
            output = build_output(segments)

            entry = {
                "messages": [
                    {"role": "user", "content": f"<audio>{prompt}"},
                    {"role": "assistant", "content": output},
                ],
                "audios": [audio_path],
            }
            entries.append(entry)
    return entries


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PROMPTS_FILE, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(prompts)} prompts.")

    for base_dir, csv_name, json_name in SPLITS:
        csv_path = os.path.join(base_dir, csv_name)
        json_path = os.path.join(OUTPUT_DIR, json_name)

        if not os.path.exists(csv_path):
            print(f"Skipping missing file: {csv_path}")
            continue

        entries = convert_csv(csv_path, prompts)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"data": entries}, f, indent=2, ensure_ascii=False)
        print(f"Wrote {len(entries)} entries → {json_path}")


if __name__ == "__main__":
    main()
