"""
Fix broken audio paths in the af3 JSON dataset.

Rules applied:
  - s3://data/FSD50K/...       -> s3://data/FSD50K/clips/...
  - s3://data/UrbanSound/...   -> s3://data/UrbanSound8K/...

Usage:
    python fix_af3_paths.py \
        --input  salmonn_data_v1.1/af3_data/audio_skills.json \
        --output salmonn_data_v1.1/af3_data/audio_skills_fixed.json
"""

import argparse
import json
import sys
import time


def fix_path(path: str) -> tuple[str, str | None]:
    """Return (fixed_path, rule_applied_or_None)."""
    if path.startswith("s3://data/FSD50K/") and "/clips/" not in path:
        return path.replace("s3://data/FSD50K/", "s3://data/FSD50K/clips/", 1), "FSD50K"
    if path.startswith("s3://data/UrbanSound/"):
        return path.replace("s3://data/UrbanSound/", "s3://data/UrbanSound8K/", 1), "UrbanSound"
    return path, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    t0 = time.time()
    print(f"Loading {args.input} …", flush=True)
    with open(args.input) as f:
        data = json.load(f)
    items = data["data"]
    print(f"  {len(items):,} items loaded in {time.time() - t0:.1f}s")

    counts = {"FSD50K": 0, "UrbanSound": 0}

    for item in items:
        fixed_audios = []
        for path in item.get("audios", []):
            new_path, rule = fix_path(path)
            if rule:
                counts[rule] += 1
            fixed_audios.append(new_path)
        item["audios"] = fixed_audios

    print(f"\nPaths fixed:")
    for rule, cnt in counts.items():
        print(f"  {rule:<20} {cnt:>8}")

    print(f"\nSaving to {args.output} …", flush=True)
    t1 = time.time()
    with open(args.output, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  Done in {time.time() - t1:.1f}s")
    print(f"Total wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
