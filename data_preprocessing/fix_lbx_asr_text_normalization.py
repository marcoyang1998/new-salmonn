#!/usr/bin/env python3
"""Fix English transcript punctuation/capitalization in the LBX ASR JSON."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, TextIO


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


class JsonStreamReader:
    def __init__(self, handle: TextIO, chunk_size: int = 1024 * 1024) -> None:
        self.handle = handle
        self.chunk_size = chunk_size
        self.decoder = json.JSONDecoder()
        self.buffer = ""
        self.eof = False

    def read_more(self) -> None:
        chunk = self.handle.read(self.chunk_size)
        if chunk:
            self.buffer += chunk
        else:
            self.eof = True

    def ensure_buffer(self) -> None:
        if not self.buffer and not self.eof:
            self.read_more()

    def skip_ws(self) -> None:
        while True:
            self.ensure_buffer()
            stripped = self.buffer.lstrip()
            if len(stripped) != len(self.buffer):
                self.buffer = stripped
                continue
            if self.buffer or self.eof:
                return

    def peek(self) -> str:
        self.skip_ws()
        return self.buffer[:1]

    def expect(self, expected: str) -> None:
        self.skip_ws()
        while len(self.buffer) < len(expected) and not self.eof:
            self.read_more()
        if not self.buffer.startswith(expected):
            raise ValueError(f"Expected {expected!r}, got {self.buffer[:20]!r}")
        self.buffer = self.buffer[len(expected) :]

    def read_value(self) -> Any:
        self.skip_ws()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer)
            except json.JSONDecodeError:
                if self.eof:
                    raise
                self.read_more()
                continue
            self.buffer = self.buffer[end:]
            return value


def update_stats_for_text(stats: Counter[str], text: str, include_mixed: bool) -> None:
    stripped = text.strip()
    if not stripped:
        stats["empty_assistant_transcripts"] += 1
        return

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


def normalize_item(
    item: dict[str, Any],
    idx: int,
    include_mixed: bool,
    stats: Counter[str],
    examples: list[tuple[int, str, str]],
    dry_run: bool,
) -> dict[str, Any]:
    for message in item.get("messages", []):
        if message.get("role") != "assistant":
            continue
        original = str(message.get("content", ""))
        update_stats_for_text(stats, original, include_mixed=include_mixed)
        if not should_fix_as_english(original, include_mixed=include_mixed):
            continue
        normalized = normalize_english_transcript(original)
        if normalized != original:
            stats["changed_transcripts"] += 1
            if len(examples) < 20:
                examples.append((idx, original, normalized))
            if not dry_run:
                message["content"] = normalized
    return item


def stream_array(
    reader: JsonStreamReader,
    output_handle: TextIO | None,
    include_mixed: bool,
    dry_run: bool,
    stats: Counter[str],
    examples: list[tuple[int, str, str]],
    start_idx: int = 0,
) -> int:
    reader.expect("[")
    if output_handle is not None:
        output_handle.write("[\n")

    idx = start_idx
    first = True
    while True:
        next_char = reader.peek()
        if next_char == "]":
            reader.expect("]")
            break
        if not first:
            reader.expect(",")

        value = reader.read_value()
        if isinstance(value, dict):
            value = normalize_item(
                value,
                idx=idx,
                include_mixed=include_mixed,
                stats=stats,
                examples=examples,
                dry_run=dry_run,
            )
        else:
            stats["non_dict_items"] += 1

        if output_handle is not None:
            if not first:
                output_handle.write(",\n")
            output_handle.write("  ")
            json.dump(value, output_handle, ensure_ascii=False)

        first = False
        idx += 1

    if output_handle is not None:
        output_handle.write("\n]")
    return idx


def stream_normalize_json(
    input_path: Path,
    output_path: Path | None,
    include_mixed: bool,
    dry_run: bool,
) -> tuple[Counter[str], list[tuple[int, str, str]]]:
    stats: Counter[str] = Counter()
    examples: list[tuple[int, str, str]] = []
    item_keys = {"data", "annotation", "samples", "items"}

    with input_path.open("r", encoding="utf-8") as input_handle:
        reader = JsonStreamReader(input_handle)
        if not dry_run and output_path is None:
            raise ValueError("output_path is required unless dry_run is set")
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = None if dry_run else output_path.open("w", encoding="utf-8")
        try:
            first_char = reader.peek()
            if first_char == "[":
                stream_array(reader, output_handle, include_mixed, dry_run, stats, examples)
                reader.skip_ws()
                if reader.buffer or not reader.eof:
                    raise ValueError("Unexpected content after top-level JSON array")
            elif first_char == "{":
                stream_object(reader, output_handle, item_keys, include_mixed, dry_run, stats, examples)
                reader.skip_ws()
                if reader.buffer or not reader.eof:
                    raise ValueError("Unexpected content after top-level JSON object")
            else:
                raise ValueError(f"Expected top-level object or array, got {first_char!r}")
        finally:
            if output_handle is not None:
                output_handle.close()

    return stats, examples


def stream_object(
    reader: JsonStreamReader,
    output_handle: TextIO | None,
    item_keys: set[str],
    include_mixed: bool,
    dry_run: bool,
    stats: Counter[str],
    examples: list[tuple[int, str, str]],
) -> None:
    reader.expect("{")
    if output_handle is not None:
        output_handle.write("{\n")

    first = True
    normalized_item_array = False
    while True:
        next_char = reader.peek()
        if next_char == "}":
            reader.expect("}")
            break
        if not first:
            reader.expect(",")

        key = reader.read_value()
        if not isinstance(key, str):
            raise ValueError(f"Expected object key string, got {type(key).__name__}")
        reader.expect(":")

        if output_handle is not None:
            if not first:
                output_handle.write(",\n")
            output_handle.write(f"  {json.dumps(key, ensure_ascii=False)}: ")

        if key in item_keys and reader.peek() == "[":
            stream_array(reader, output_handle, include_mixed, dry_run, stats, examples)
            normalized_item_array = True
        else:
            value = reader.read_value()
            if output_handle is not None:
                json.dump(value, output_handle, ensure_ascii=False)

        first = False

    if not normalized_item_array:
        raise ValueError(f"Could not find a top-level item array under one of {sorted(item_keys)}")

    if output_handle is not None:
        output_handle.write("\n}\n")


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

    out_path = args.input if args.in_place else args.output or output_path_for(args.input)
    write_path = out_path
    if args.in_place and not args.dry_run:
        write_path = args.input.with_name(f".{args.input.name}.tmp")

    stats, examples = stream_normalize_json(
        input_path=args.input,
        output_path=None if args.dry_run else write_path,
        include_mixed=args.include_mixed,
        dry_run=args.dry_run,
    )
    print_report(stats, examples)

    if args.dry_run:
        return

    if args.in_place:
        os.replace(write_path, args.input)
    print(f"\nWrote normalized JSON to {out_path}")


if __name__ == "__main__":
    main()
