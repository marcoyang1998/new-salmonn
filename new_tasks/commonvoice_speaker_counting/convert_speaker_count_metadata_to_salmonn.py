#!/usr/bin/env python3
"""Convert mixed speaker-counting metadata into SALMONN-style training JSON."""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_METADATA = Path("salmonn_data_v1.1/speaker_counting/metadata_2to6_maxutt10.json")
DEFAULT_OUT = Path(
    "salmonn_data_v1.1/speaker_counting/speaker_counting_train_salmonn_10k_2to6_maxutt10.json"
)
SPEAKER_COUNTING_PROMPTS = [
    "Count the number of speakers in this audio.",
    "Determine the number of people speaking in this audio clip.",
    "How many different people are speaking in this audio clip?",
    "How many distinct speakers are there in this audio?",
    "How many unique speakers can be heard in this recording?",
    "How many speakers can you identify in this recording?",
]


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unwrap_data(payload: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("samples", "data", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find a list of records in {path}")


def first_present(item: Dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value is not None:
            return value
    return None


def build_salmonn_item(sample: Dict[str, Any], prompt: str, answer_suffix: str) -> Dict[str, Any]:
    audio_path = first_present(sample, ("mixed_audio_path", "audio_path", "wav_path", "path"))
    num_speakers = first_present(sample, ("num_speakers", "speaker_count", "count"))
    duration = first_present(sample, ("duration_sec", "duration", "dur"))

    if audio_path is None:
        raise ValueError(f"Missing audio path in sample: {sample.get('sample_id', '<unknown>')}")
    if num_speakers is None:
        raise ValueError(f"Missing speaker count in sample: {sample.get('sample_id', '<unknown>')}")
    if duration is None:
        raise ValueError(f"Missing duration in sample: {sample.get('sample_id', '<unknown>')}")

    answer = f"{num_speakers}{answer_suffix}"
    return {
        "messages": [
            {"role": "user", "content": f"<audio>{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        "audios": [str(audio_path)],
        "durations": [float(duration)],
    }


def convert(metadata_path: Path, out_path: Path, seed: int, answer_suffix: str) -> None:
    metadata = read_json(metadata_path)
    samples = unwrap_data(metadata, metadata_path)
    rng = random.Random(seed)

    data = [
        build_salmonn_item(sample, rng.choice(SPEAKER_COUNTING_PROMPTS), answer_suffix)
        for sample in samples
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"data": data}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = collections.Counter(item["messages"][1]["content"] for item in data)
    prompt_counts = collections.Counter(item["messages"][0]["content"] for item in data)
    print(f"Wrote {len(data)} items to {out_path}")
    print("Speaker-count labels:")
    for label, count in sorted(counts.items(), key=lambda kv: int(str(kv[0]).split()[0])):
        print(f"  {label}: {count}")
    print(f"Prompt variants used: {len(prompt_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--answer_suffix",
        default="",
        help="Optional suffix for answers, e.g. ' people'. Default keeps bare numeric labels.",
    )
    args = parser.parse_args()

    convert(args.metadata, args.out, args.seed, args.answer_suffix)


if __name__ == "__main__":
    main()
