#!/usr/bin/env python3
"""Print duration bucket statistics for a JSON dataset file."""

import json
import sys
from collections import Counter

BUCKETS = [
    ("0-10s",  0,  10),
    ("10-20s", 10, 20),
    ("20-30s", 20, 30),
    ("30-40s", 30, 40),
    ("40-50s", 40, 50),
    ("50-60s", 50, 60),
    (">60s",   60, float("inf")),
]

def get_bucket(dur):
    for label, lo, hi in BUCKETS:
        if lo < dur <= hi or (lo == 0 and dur <= hi):
            return label
    return ">60s"

def main(path):
    counts = Counter()
    total = 0
    total_duration = 0.0

    # Detect format by reading the first non-empty line
    with open(path) as f:
        first_line = f.readline().strip()

    if first_line in ("{", '{"data": [', '{ "data": ['):
        # JSON with "data" wrapper — may be compact or pretty-printed
        # Peek at line after header to distinguish
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if stripped and stripped not in ("{", '"data": [', '{"data": ['):
                    second_content = stripped
                    break
        if second_content.startswith('{'):
            # Compact: one item per line, skip first header line
            with open(path) as f:
                f.readline()
                for line in f:
                    line = line.strip().rstrip(",")
                    if not line or line in ("]}",  "]"):
                        break
                    item = json.loads(line)
                    dur = item["durations"][0] if item.get("durations") else 0
                    counts[get_bucket(dur)] += 1
                    total += 1
                    total_duration += dur
        else:
            # Pretty-printed: load whole file
            with open(path) as f:
                data = json.load(f)
            for item in data["data"]:
                dur = item["durations"][0] if item.get("durations") else 0
                counts[get_bucket(dur)] += 1
                total += 1
                total_duration += dur
    else:
        # Pure JSONL
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                dur = item["durations"][0] if item.get("durations") else 0
                counts[get_bucket(dur)] += 1
                total += 1
                total_duration += dur

    print(f"File: {path}")
    print(f"Total items: {total}\n")
    print(f"Total duration: {total_duration / 3600:.2f} h ({total_duration:.1f} s)\n")
    print(f"{'Bucket':<10} {'Count':>10} {'Percentage':>12}")
    print("-" * 34)
    for label, _, _ in BUCKETS:
        n = counts[label]
        pct = 100 * n / total if total else 0
        print(f"{label:<10} {n:>10,} {pct:>11.1f}%")
    print("-" * 34)
    print(f"{'Total':<10} {total:>10,} {'100.0%':>12}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_json>")
        sys.exit(1)
    main(sys.argv[1])
