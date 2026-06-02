"""
Convert natural-language timestamp annotations to SALMONN timestamp tokens.

Token spec:
  - 601 tokens covering 0.00 s – 60.00 s in 0.10 s steps
  - Token format:  <|{value:.2f}|>   e.g. <|0.00|> <|0.60|> <|60.00|>

Input JSON format (assistant content):
    "[[0.6, 5.68], [3.25, 4.85]]"

Output JSON format (assistant content):
    "<|0.60|> <|5.70|>, <|3.30|> <|4.90|>"

Usage:
    python convert_timestamps_to_tokens.py input.json output.json
"""

import sys
import json
import re
import ast

# ── token parameters ──────────────────────────────────────────────────────────
STEP = 0.1          # granularity in seconds
MIN_T = 0.0         # minimum timestamp (s)
MAX_T = 60.0        # maximum timestamp (s)
NUM_TOKENS = 601    # 0.0, 0.1, ..., 60.0
# ─────────────────────────────────────────────────────────────────────────────


def snap(value: float) -> float:
    """Round *value* to the nearest token boundary and clamp to [MIN_T, MAX_T]."""
    snapped = round(round(value / STEP) * STEP, 10)   # avoid float drift
    return max(MIN_T, min(MAX_T, snapped))


def to_token(value: float) -> str:
    return f"<|{snap(value):.2f}|>"


# Regex that matches a JSON-style list of [a, b] pairs:
# [[a1, b1], [a2, b2], ...]  or  [a, b]  (single pair)
_INTERVALS_RE = re.compile(
    r"\[\s*\[.*?\]\s*\]"   # [[...]]
    r"|\[\s*[\d.]+\s*,\s*[\d.]+\s*\]",  # bare [a, b]
    re.DOTALL,
)


def convert_intervals(intervals: list) -> str:
    """Convert a list of [a, b] pairs to token format string."""
    parts = [f"[{to_token(float(a))}, {to_token(float(b))}]" for a, b in intervals]
    return "[" + ", ".join(parts) + "]"


def convert_content(text: str) -> str | None:
    """
    Parse *text* and replace all timestamps with tokens.

    Supported formats:
      - List of pairs:  [[a1, b1], [a2, b2], ...]  or bare [a, b]
      - Dict of label -> intervals:  {"label": [[a1, b1], ...], ...}

    Returns None if the content cannot be parsed as either format.
    """
    text = text.strip()

    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None

    # ── dict format: {"label": [[a, b], ...], ...} ───────────────────────────
    if isinstance(parsed, dict):
        out = {}
        for label, intervals in parsed.items():
            # Normalise single pair [a, b] to [[a, b]]
            if (
                isinstance(intervals, list)
                and len(intervals) == 2
                and isinstance(intervals[0], (int, float))
            ):
                intervals = [intervals]
            if not (
                isinstance(intervals, list)
                and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in intervals)
            ):
                return None  # unexpected structure
            out[label] = convert_intervals(intervals)
        # Reconstruct as a dict literal with token strings as values
        inner = ", ".join(f'"{k}": {v}' for k, v in out.items())
        return "{" + inner + "}"

    # ── list format: [[a, b], ...] or [a, b] ─────────────────────────────────
    if isinstance(parsed, list):
        # Normalise bare [a, b] to [[a, b]]
        if len(parsed) == 2 and isinstance(parsed[0], (int, float)):
            parsed = [parsed]
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 for p in parsed):
            return None
        return convert_intervals(parsed)

    return None


def convert_file(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data if isinstance(data, list) else data.get("data", data)

    n_converted = 0
    n_skipped = 0

    for entry in entries:
        for msg in entry.get("messages", []):
            if msg.get("role") != "assistant":
                continue
            original = msg["content"]
            converted = convert_content(original)
            if converted is not None:
                msg["content"] = converted
                n_converted += 1
            else:
                n_skipped += 1

    # Preserve top-level wrapper key if present
    if isinstance(data, dict) and "data" in data:
        out_obj = {**data, "data": entries}
    else:
        out_obj = entries

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(
        f"Done. Converted {n_converted} assistant turn(s), "
        f"skipped {n_skipped} (no parseable intervals). "
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_timestamps_to_tokens.py <input.json> <output.json>")
        sys.exit(1)
    convert_file(sys.argv[1], sys.argv[2])
