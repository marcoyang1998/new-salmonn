#!/usr/bin/env python3
"""Convert SPGISpeech metadata into SALMONN-style ASR training JSON."""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_METADATA = Path("salmonn_data_v1.1/spgispeech_s_metadata.json")
DEFAULT_OUT = Path("salmonn_data_v1.1/spgispeech_s_asr_train_salmonn.json")

ASR_PROMPTS = [
    "<audio>Can you recognize what you heard in the speech?",
    "<audio>Can you transcribe the speech into a written format?",
    "<audio>Can you write down the transcription of the speech?",
    "<audio>Give me the transcription of the speech you heard.",
    "<audio>Listen to the speech and recognize its content.",
    "<audio>Listen to the speech and write down its content.",
    "<audio>Please help me to transcribe the speech into a written format.",
    "<audio>Please transcribe the speech into a written format.",
    "<audio>Please write down the transcription of the speech.",
    "<audio>Put the speech into a written format.",
    "<audio>Recognize the content of the speech you heard.",
    "<audio>Recognize the speech and give me the transcription.",
    "<audio>Recognize the speech and write it down in a written format.",
    "<audio>What is the content of the speech you heard?",
    "<audio>Write down the content of the speech you heard.",
]

ALLOWED_PUNCTUATION = {",", ".", "!", "?", "'"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unwrap_records(payload: Any, path: Path) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "samples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError(f"Could not find a list of records in {path}")


def normalize_transcript(text: Any) -> str:
    text = str(text).strip()
    text = text.replace("--", "").replace('"', "")

    kept_chars = []
    for char in text:
        if char.isalnum() or char.isspace() or char in ALLOWED_PUNCTUATION:
            kept_chars.append(char)

    normalized = re.sub(r"\s+", " ", "".join(kept_chars)).strip()
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


def audio_path_for_record(wav_path: Any, metadata_path: Path, keep_metadata_relative: bool) -> str:
    audio_path = Path(str(wav_path))
    if audio_path.is_absolute() or not keep_metadata_relative:
        return str(audio_path)
    return str(metadata_path.parent / audio_path)


def build_salmonn_item(
    record: Dict[str, Any],
    prompt: str,
    metadata_path: Path,
    keep_metadata_relative: bool,
) -> Dict[str, Any]:
    if "wav_path" not in record:
        raise ValueError(f"Missing wav_path in record: {record}")
    if "duration" not in record:
        raise ValueError(f"Missing duration in record: {record}")

    transcript = normalize_transcript(record.get("transcript", ""))
    if not transcript:
        raise ValueError("Normalized transcript is empty")

    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": transcript},
        ],
        "audios": [
            audio_path_for_record(
                record["wav_path"],
                metadata_path=metadata_path,
                keep_metadata_relative=keep_metadata_relative,
            )
        ],
        "durations": [float(record["duration"])],
        "task_type": "asr",
    }


def convert(
    metadata_path: Path,
    out_path: Path,
    seed: int,
    keep_metadata_relative: bool,
) -> None:
    records = unwrap_records(read_json(metadata_path), metadata_path)
    rng = random.Random(seed)

    data = []
    skipped_empty = 0
    for record in records:
        prompt = rng.choice(ASR_PROMPTS)
        try:
            data.append(
                build_salmonn_item(
                    record,
                    prompt=prompt,
                    metadata_path=metadata_path,
                    keep_metadata_relative=keep_metadata_relative,
                )
            )
        except ValueError as exc:
            if str(exc) == "Normalized transcript is empty":
                skipped_empty += 1
                continue
            raise

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"data": data}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    prompt_counts = collections.Counter(item["messages"][0]["content"] for item in data)
    print(f"Read {len(records)} records from {metadata_path}")
    print(f"Wrote {len(data)} SALMONN ASR items to {out_path}")
    if skipped_empty:
        print(f"Skipped {skipped_empty} records with empty normalized transcripts")
    print("Prompt counts:")
    for prompt, count in sorted(prompt_counts.items()):
        print(f"  {count:6d}  {prompt}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--keep_wav_path_as_is",
        action="store_true",
        help="Keep metadata wav_path values unchanged instead of resolving them relative to the metadata file.",
    )
    args = parser.parse_args()

    convert(
        metadata_path=args.metadata,
        out_path=args.out,
        seed=args.seed,
        keep_metadata_relative=not args.keep_wav_path_as_is,
    )


if __name__ == "__main__":
    main()
