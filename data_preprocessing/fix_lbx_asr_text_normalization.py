#!/usr/bin/env python3
"""Fix English transcript punctuation/capitalization in the LBX ASR JSON."""

from __future__ import annotations

import argparse
import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("salmonn_data_v1.1/lbx_asr/lbx_asr_20k_with_durations.json")

LATIN_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[\u3400-\u9fff]")
SENTENCE_START_RE = re.compile(r"(^|[.!?]\s+)([a-z])")
PUNCT_SPACING_RE = re.compile(r"([.!?])(?=[A-Za-z0-9])")
COMMA_SPACING_RE = re.compile(r",(?=[A-Za-z0-9])")
LOWERCASE_I_RE = re.compile(r"\bi\b")

ENGLISH_PUNCTUATION_MAP = str.maketrans(
    {
        "\u3002": ".",
        "\uff0c": ",",
        "\uff1f": "?",
        "\uff01": "!",
    }
)


def unwrap_items(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("data", "annotation", "samples", "items"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise ValueError("Could not find a list of records in the input JSON")


def has_latin(text: str) -> bool:
    return bool(LATIN_RE.search(text))


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text))


def should_fix_as_english(text: str, include_mixed: bool) -> bool:
    if not has_latin(text):
        return False
    return include_mixed or not has_cjk(text)


def capitalize_sentence_starts(text: str) -> str:
    return SENTENCE_START_RE.sub(lambda match: match.group(1) + match.group(2).upper(), text)


def normalize_english_transcript(text: str) -> str:
    normalized = text.translate(ENGLISH_PUNCTUATION_MAP)
    normalized = PUNCT_SPACING_RE.sub(r"\1 ", normalized)
    normalized = COMMA_SPACING_RE.sub(", ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = capitalize_sentence_starts(normalized)
    normalized = LOWERCASE_I_RE.sub("I", normalized)
    return normalized


def iter_assistant_messages(items: Iterable[dict[str, Any]]) -> Iterable[tuple[int, dict[str, Any]]]:
    for idx, item in enumerate(items):
        for message in item.get("messages", []):
            if message.get("role") == "assistant":
                yield idx, message


def output_path_for(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_normalized{input_path.suffix}")


def summarize(items: list[dict[str, Any]], include_mixed: bool) -> Counter[str]:
    stats: Counter[str] = Counter()
    for _idx, message in iter_assistant_messages(items):
        text = str(message.get("content", ""))
        stripped = text.strip()
        if not stripped:
            stats["empty_assistant_transcripts"] += 1
            continue

        latin = has_latin(stripped)
        cjk = has_cjk(stripped)
        if latin and cjk:
            stats["mixed_latin_cjk_transcripts"] += 1
        elif latin:
            stats["english_only_transcripts"] += 1
        elif cjk:
            stats["chinese_only_transcripts"] += 1
        else:
            stats["other_transcripts"] += 1

        if should_fix_as_english(stripped, include_mixed):
            stats["selected_for_english_normalization"] += 1
            if stripped.endswith("\u3002"):
                stats["selected_end_with_ideographic_period"] += 1
            if "\uff0c" in stripped:
                stats["selected_have_fullwidth_comma"] += 1
            if "\uff1f" in stripped:
                stats["selected_have_fullwidth_question"] += 1
            if "\uff01" in stripped:
                stats["selected_have_fullwidth_exclamation"] += 1
            first_alpha = next((char for char in stripped if char.isalpha()), "")
            if first_alpha and first_alpha.islower():
                stats["selected_lowercase_initial"] += 1
    return stats


def normalize_payload(payload: Any, include_mixed: bool, dry_run: bool) -> tuple[Any, Counter[str], list[tuple[int, str, str]]]:
    output = payload if dry_run else copy.deepcopy(payload)
    items = unwrap_items(output)
    stats = summarize(items, include_mixed=include_mixed)
    examples: list[tuple[int, str, str]] = []

    for idx, message in iter_assistant_messages(items):
        original = str(message.get("content", ""))
        if not should_fix_as_english(original, include_mixed=include_mixed):
            continue
        normalized = normalize_english_transcript(original)
        if normalized != original:
            stats["changed_transcripts"] += 1
            if len(examples) < 20:
                examples.append((idx, original, normalized))
            if not dry_run:
                message["content"] = normalized

    return output, stats, examples


def print_report(stats: Counter[str], examples: list[tuple[int, str, str]]) -> None:
    print("LBX ASR normalization report")
    for key in sorted(stats):
        print(f"{key}: {stats[key]}")
    if examples:
        print("\nExamples:")
        for idx, before, after in examples:
            print(f"[{idx}] {before!r} -> {after!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-mixed",
        action="store_true",
        help="Also normalize transcripts that contain both Latin and CJK characters.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print the audit report; do not write output.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite the input JSON.")
    args = parser.parse_args()

    if args.output and args.in_place:
        raise ValueError("--output and --in-place cannot be used together")

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    output, stats, examples = normalize_payload(
        payload,
        include_mixed=args.include_mixed,
        dry_run=args.dry_run,
    )
    print_report(stats, examples)

    if args.dry_run:
        return

    out_path = args.input if args.in_place else args.output or output_path_for(args.input)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"\nWrote normalized JSON to {out_path}")


if __name__ == "__main__":
    main()
