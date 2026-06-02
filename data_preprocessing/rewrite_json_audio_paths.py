"""Rewrite audio paths in a SALMONN JSON file.

Currently supported dataset rules:
  - GigaSpeech:
    /mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data
    -> s3://data/GigaSpeech/gigaspeech/data

Usage:
    python data_preprocessing/rewrite_json_audio_paths.py \
        --input-json salmonn_data_v1.1/contextual_biasing/input.json \
        --output-dir salmonn_data_v1.1/contextual_biasing_s3
"""

import argparse
import json
import time
from pathlib import Path


GIGASPEECH_OLD_PREFIX = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data"
GIGASPEECH_NEW_PREFIX = "s3://data/GigaSpeech/gigaspeech/data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite dataset-specific audio paths in a JSON file.")
    parser.add_argument("--input-json", "-i", required=True, help="Input JSON file path.")
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Output directory. The rewritten JSON keeps the same file name as the input.",
    )
    return parser.parse_args()


def detect_dataset(audio_path: str) -> str | None:
    audio_path_lower = audio_path.lower()
    if "gigaspeech" in audio_path_lower:
        return "gigaspeech"
    return None


def rewrite_gigaspeech_path(audio_path: str) -> tuple[str, bool]:
    if audio_path.startswith(GIGASPEECH_OLD_PREFIX):
        return audio_path.replace(GIGASPEECH_OLD_PREFIX, GIGASPEECH_NEW_PREFIX, 1), True
    return audio_path, False


def rewrite_audio_path(audio_path: str) -> tuple[str, str | None, bool]:
    dataset = detect_dataset(audio_path)
    if dataset == "gigaspeech":
        new_path, changed = rewrite_gigaspeech_path(audio_path)
        return new_path, dataset, changed
    return audio_path, dataset, False


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    output_path = output_dir / input_path.name

    t0 = time.time()
    print(f"Loading {input_path} ...", flush=True)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("data", [])
    print(f"  {len(items):,} items loaded in {time.time() - t0:.1f}s")

    stats = {
        "items": len(items),
        "audio_paths_seen": 0,
        "audio_paths_rewritten": 0,
        "gigaspeech_paths_rewritten": 0,
        "unsupported_dataset_paths": 0,
    }

    for item in items:
        audios = item.get("audios", [])
        rewritten_audios = []
        for audio_path in audios:
            stats["audio_paths_seen"] += 1
            new_path, dataset, changed = rewrite_audio_path(audio_path)
            if changed:
                stats["audio_paths_rewritten"] += 1
                if dataset == "gigaspeech":
                    stats["gigaspeech_paths_rewritten"] += 1
            elif dataset is not None:
                stats["unsupported_dataset_paths"] += 1
            rewritten_audios.append(new_path)
        item["audios"] = rewritten_audios

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to {output_path} ...", flush=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print("\nRewrite summary:")
    print(f"  items:                       {stats['items']:>8}")
    print(f"  audio paths seen:           {stats['audio_paths_seen']:>8}")
    print(f"  audio paths rewritten:      {stats['audio_paths_rewritten']:>8}")
    print(f"  gigaspeech paths rewritten: {stats['gigaspeech_paths_rewritten']:>8}")
    print(f"  unsupported dataset paths:  {stats['unsupported_dataset_paths']:>8}")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()