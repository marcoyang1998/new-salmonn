#!/usr/bin/env python3
"""Convert WenetSpeech Lhotse cuts JSONL(.gz) to SALMONN-style ASR JSON."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import random
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_MANIFEST = Path("wenetspeech_cuts_S.jsonl.gz")
DEFAULT_OUTPUT = Path("salmonn_data_v1.1/wenetspeech/wenetspeech_S_asr_salmonn.json")

ASR_PROMPTS = [
    "<audio>你能识别刚才听到的语音内容吗？",
    "<audio>请写下你听到的语音内容。",
    "<audio>请写下这段语音的转写结果。",
    "<audio>请将语音中的内容写下来。",
    "<audio>请把你听到的语音转写出来。",
    "<audio>请识别你听到的语音内容。",
]


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def count_lines(path: Path) -> int:
    with open_text(path, "rt") as handle:
        return sum(1 for _ in handle)


def maybe_rewrite_audio_path(
    audio_path: str,
    audio_root: str | None,
    strip_prefix: str | None,
) -> str:
    if strip_prefix and audio_path.startswith(strip_prefix):
        audio_path = audio_path[len(strip_prefix) :].lstrip("/")
    if audio_root:
        return str(Path(audio_root) / audio_path)
    return audio_path


def first_audio_path(record: dict[str, Any]) -> str:
    sources = record.get("recording", {}).get("sources", [])
    if not sources:
        raise ValueError("recording.sources is empty")
    source = sources[0].get("source")
    if not source:
        raise ValueError("recording.sources[0].source is empty")
    return str(source)


def first_text(record: dict[str, Any]) -> str:
    supervisions = record.get("supervisions", [])
    if not supervisions:
        raise ValueError("supervisions is empty")
    text = supervisions[0].get("text", "")
    text = str(text).strip().replace('"', "")
    if not text:
        raise ValueError("supervisions[0].text is empty")
    return text


def duration(record: dict[str, Any]) -> float:
    value = record.get("duration", record.get("recording", {}).get("duration"))
    if value is None:
        raise ValueError("duration is missing")
    return float(value)


def build_item(
    record: dict[str, Any],
    prompt: str,
    audio_root: str | None,
    strip_prefix: str | None,
    include_task_type: bool,
) -> dict[str, Any]:
    item = {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": first_text(record)},
        ],
        "audios": [
            maybe_rewrite_audio_path(
                first_audio_path(record),
                audio_root=audio_root,
                strip_prefix=strip_prefix,
            )
        ],
        "durations": [duration(record)],
    }
    if include_task_type:
        item["task_type"] = "asr"
    return item


def convert(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    total = None if args.no_count else count_lines(args.manifest)

    data = []
    skipped = collections.Counter()
    prompt_counts = collections.Counter()

    with open_text(args.manifest, "rt") as handle:
        if tqdm is not None:
            iterator = tqdm(
                handle,
                total=total,
                unit="cut",
                dynamic_ncols=True,
                desc="converting",
            )
        else:
            iterator = handle
        for line_idx, line in enumerate(iterator, start=1):
            if args.limit is not None and len(data) >= args.limit:
                break
            try:
                record = json.loads(line)
                dur = duration(record)
                if args.min_duration is not None and dur < args.min_duration:
                    skipped["too_short"] += 1
                    continue
                if args.max_duration is not None and dur > args.max_duration:
                    skipped["too_long"] += 1
                    continue
                prompt = rng.choice(ASR_PROMPTS)
                data.append(
                    build_item(
                        record,
                        prompt=prompt,
                        audio_root=args.audio_root,
                        strip_prefix=args.strip_prefix,
                        include_task_type=args.include_task_type,
                    )
                )
                prompt_counts[prompt] += 1
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                skipped[type(exc).__name__] += 1
                if args.strict:
                    raise RuntimeError(f"Failed on line {line_idx}: {exc}") from exc

            if tqdm is not None and len(data) % 1000 == 0:
                iterator.set_postfix(written=len(data), skipped=sum(skipped.values()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"data": data}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Read manifest: {args.manifest}")
    print(f"Wrote {len(data)} SALMONN ASR items to {args.output}")
    if skipped:
        print("Skipped records:")
        for reason, count in sorted(skipped.items()):
            print(f"  {reason}: {count}")
    print("Prompt counts:")
    for prompt, count in sorted(prompt_counts.items()):
        print(f"  {count:6d}  {prompt}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-duration", type=float, default=None)
    parser.add_argument("--max-duration", type=float, default=None)
    parser.add_argument(
        "--audio-root",
        default=None,
        help="Optional root to prepend to each cut recording source path.",
    )
    parser.add_argument(
        "--strip-prefix",
        default=None,
        help="Optional prefix to strip from each cut recording source before prepending --audio-root.",
    )
    parser.add_argument(
        "--include-task-type",
        action="store_true",
        help='Add "task_type": "asr" to each item. Disabled by default to match the LBX reference file.',
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop on the first malformed record instead of skipping it.",
    )
    parser.add_argument(
        "--no-count",
        action="store_true",
        help="Skip the initial line-count pass for faster startup.",
    )
    return parser.parse_args()


def main() -> None:
    convert(parse_args())


if __name__ == "__main__":
    main()
