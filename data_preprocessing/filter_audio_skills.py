"""
Filter audio_skills JSON dataset by removing problematic items:
  1. Empty assistant response
  2. Empty user prompt (content is just <audio> with no question)
  3. MC: (A) present but (B) missing (incomplete single-option MC)
  4. MC: non-consecutive options (e.g. (A)(C) with no (B))
  5. MC: answer letter not among the options listed in the prompt

Usage:
    python filter_audio_skills.py --input <input.json> --output <output.json>
"""

import argparse
import json
import re
from decimal import Decimal

import ijson


class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Decimal):
            return float(o)
        return super().default(o)


OPTION_PATTERN = re.compile(r"\(([A-Z])\)")


def detect_mc_options(content):
    return sorted(set(OPTION_PATTERN.findall(content)))


def is_consecutive(letters):
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    return letters == expected


def get_issues(item):
    issues = []
    messages = item.get("messages") or []

    user_content = ""
    assistant_content = ""
    for m in messages:
        if m.get("role") == "user":
            user_content = m.get("content") or ""
        elif m.get("role") == "assistant":
            assistant_content = m.get("content") or ""

    # 1. Empty assistant response
    if not assistant_content:
        issues.append("empty_assistant")

    # 2. Empty user prompt (only <audio>, no question)
    if not user_content.replace("<audio>", "").strip():
        issues.append("empty_user_prompt")

    # MC checks
    options = detect_mc_options(user_content)
    if options:
        # 3. (A) present but (B) missing
        if "A" in options and "B" not in options:
            issues.append("mc_only_option_A")

        # 4. Non-consecutive options
        if not is_consecutive(options):
            issues.append("mc_non_consecutive_options")

        # 5. Answer letter not among listed options
        if assistant_content:
            raw = assistant_content.strip().strip(".")
            m = re.match(r"^\(?([A-Z])\)?", raw)
            if m and m.group(1) not in options:
                issues.append("mc_answer_letter_out_of_range")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output JSON file")
    args = parser.parse_args()

    from collections import defaultdict
    removed_counts = defaultdict(int)
    total = 0
    kept = 0

    with open(args.input, "rb") as fin, open(args.output, "w") as fout:
        fout.write('{"data": [\n')
        first = True
        for item in ijson.items(fin, "data.item"):
            total += 1
            issues = get_issues(item)
            if issues:
                for iss in issues:
                    removed_counts[iss] += 1
                continue
            if not first:
                fout.write(",\n")
            fout.write(json.dumps(item, cls=DecimalEncoder))
            first = False
            kept += 1
            if total % 500000 == 0:
                print(f"  ...processed {total:,}, kept {kept:,}", flush=True)
        fout.write("\n]}\n")

    print(f"\nDone.")
    print(f"  Total scanned                  : {total:,}")
    for key, cnt in sorted(removed_counts.items(), key=lambda x: -x[1]):
        print(f"  Removed ({key:<30}): {cnt:,}")
    print(f"  Kept                           : {kept:,}")
    print(f"  Output                         : {args.output}")


if __name__ == "__main__":
    main()
