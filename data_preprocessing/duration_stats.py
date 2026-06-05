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

    with open(path) as f:
        first_line = f.readline().strip()
        if first_line.startswith('{"data"'):
            # JSON format: one item per line
            for line in f:
                line = line.strip().rstrip(",")
                if not line or line in ("]}",  "]"):
                    break
                item = json.loads(line)
                dur = item["durations"][0] if item.get("durations") else 0
                counts[get_bucket(dur)] += 1
                total += 1
        else:
            # Pure JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                dur = item["durations"][0] if item.get("durations") else 0
                counts[get_bucket(dur)] += 1
                total += 1

    print(f"File: {path}")
    print(f"Total items: {total}\n")
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
