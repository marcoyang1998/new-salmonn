"""
Split a JSON dataset file into train and test splits.

Usage:
    python split_train_test.py \
        --input  /path/to/data.json \
        --output /path/to/output_dir \
        --test_ratio 0.1 \
        [--seed 42] \
        [--strict_split]

The input JSON must have the structure: {"data": [...]}
The outputs will be named after the input file basename, e.g. for audioset_strong_qa.json:
    <output_dir>/audioset_strong_qa_train.json
    <output_dir>/audioset_strong_qa_test.json
both with the same {"data": [...]} structure.

With --strict_split, the split is done at the audio-file level so that no audio
file appears in both train and test (useful when multiple samples share the same
audio file).
"""

import argparse
import json
import os
import random
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Split a JSON dataset into train/test splits.")
    parser.add_argument("--input", required=True, help="Path to the input JSON file.")
    parser.add_argument(
        "--output", required=True, help="Directory where train.json and test.json will be written."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of data to use for the test split (default: 0.1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--strict_split",
        action="store_true",
        help=(
            "If set, split at the audio-file level so that no audio file appears in "
            "both train and test. The target test_ratio is applied to the number of "
            "unique audio files."
        ),
    )
    return parser.parse_args()


def _get_audio_keys(item):
    """Return a frozenset of audio paths for a single data item."""
    audios = item.get("audios", [])
    if isinstance(audios, str):
        audios = [audios]
    return frozenset(audios)


def strict_split(items, test_ratio, seed):
    """Split items ensuring no audio file overlap between train and test."""
    # Build a mapping: unique audio file -> list of item indices that use it
    audio_to_indices = defaultdict(list)
    for idx, item in enumerate(items):
        for audio in _get_audio_keys(item):
            audio_to_indices[audio].append(idx)

    unique_audios = list(audio_to_indices.keys())
    random.seed(seed)
    random.shuffle(unique_audios)

    n_test_audios = max(1, int(len(unique_audios) * test_ratio))
    test_audios = set(unique_audios[:n_test_audios])
    train_audios = set(unique_audios[n_test_audios:])

    # An item belongs to test only if ALL its audios are in test_audios and
    # NONE are in train_audios (handles multi-audio samples gracefully).
    test_items, train_items = [], []
    for item in items:
        keys = _get_audio_keys(item)
        if keys and keys <= test_audios:
            test_items.append(item)
        else:
            train_items.append(item)

    return train_items, test_items


def main():
    args = parse_args()

    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError(f"--test_ratio must be between 0 and 1 (exclusive), got {args.test_ratio}")

    # Load data
    print(f"Loading data from {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    if "data" not in dataset:
        raise KeyError("Expected the JSON file to have a top-level 'data' key.")

    items = dataset["data"]
    total = len(items)
    print(f"Total samples: {total}")

    # Shuffle and split
    if args.strict_split:
        print("Using strict audio-level split (no audio file overlap between splits).")
        train_items, test_items = strict_split(items, args.test_ratio, args.seed)
    else:
        random.seed(args.seed)
        indices = list(range(total))
        random.shuffle(indices)

        n_test = max(1, int(total * args.test_ratio))
        test_indices = set(indices[:n_test])
        train_items = [items[i] for i in range(total) if i not in test_indices]
        test_items = [items[i] for i in range(total) if i in test_indices]

    print(f"Train samples: {len(train_items)}, Test samples: {len(test_items)}")

    # Write outputs
    os.makedirs(args.output, exist_ok=True)

    basename = os.path.splitext(os.path.basename(args.input))[0]
    train_path = os.path.join(args.output, f"{basename}_train.json")
    test_path = os.path.join(args.output, f"{basename}_test.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump({"data": train_items}, f, ensure_ascii=False, indent=2)
    print(f"Train split written to {train_path}")

    with open(test_path, "w", encoding="utf-8") as f:
        json.dump({"data": test_items}, f, ensure_ascii=False, indent=2)
    print(f"Test split written to {test_path}")


if __name__ == "__main__":
    main()
