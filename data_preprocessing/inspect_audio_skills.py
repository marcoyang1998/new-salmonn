"""
Comprehensive quality inspection of audio_skills JSON dataset.

Checks for:
  1.  Empty assistant response
  2.  Empty user prompt (after stripping <audio>)
  3.  No <audio> placeholder in user prompt
  4.  No messages field, or fewer than 2 messages
  5.  Wrong message order (assistant before user, or missing either role)
  6.  Empty audios list
  7.  MC: (A) present but (B) missing  [already known]
  8.  MC: non-consecutive options (e.g. (A)(C) with no (B))
  9.  MC: options present in prompt but assistant answer is not a valid letter
  10. MC: answer letter exceeds number of listed options
  11. Very short assistant response (<=2 chars) that is NOT a valid MC answer letter

Usage:
    python inspect_audio_skills.py --input <input.json> --output <report.txt>
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
    """Return sorted list of option letters found in content, e.g. ['A','B','C']."""
    return sorted(set(OPTION_PATTERN.findall(content)))


def is_consecutive(letters):
    """Check that letters form a consecutive sequence starting from A."""
    expected = [chr(ord("A") + i) for i in range(len(letters))]
    return letters == expected


def check_item(item):
    issues = []
    messages = item.get("messages")

    # 4. Missing or too-short messages list
    if not messages or len(messages) < 2:
        issues.append("missing_or_short_messages")
        return issues  # can't do further checks

    # 5. Role order
    roles = [m.get("role") for m in messages]
    if roles[0] != "user":
        issues.append("first_message_not_user")
    if roles[-1] != "assistant":
        issues.append("last_message_not_assistant")
    if "user" not in roles:
        issues.append("no_user_message")
    if "assistant" not in roles:
        issues.append("no_assistant_message")

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

    # 3. No <audio> placeholder
    if "<audio>" not in user_content:
        issues.append("no_audio_placeholder")

    # 2. Empty user prompt (after stripping <audio> tag and whitespace)
    stripped_prompt = user_content.replace("<audio>", "").strip()
    if not stripped_prompt:
        issues.append("empty_user_prompt")

    # 6. Empty audios list
    audios = item.get("audios")
    if not audios:
        issues.append("empty_audios")

    # MC checks
    options = detect_mc_options(user_content)
    if options:
        # 7. Has (A) but no (B)
        if "A" in options and "B" not in options:
            issues.append("mc_only_option_A")

        # 8. Non-consecutive options
        if not is_consecutive(options):
            issues.append(f"mc_non_consecutive_options:{','.join(options)}")

        # 9 & 10. Assistant answer validity
        if assistant_content:
            # Strip surrounding punctuation/whitespace to get the raw answer
            raw = assistant_content.strip().strip(".")
            # Accept bare letter or "(X)" or "(X) text" formats
            letter_match = re.match(r"^\(?([A-Z])\)?", raw)
            if letter_match:
                answer_letter = letter_match.group(1)
                # 10. Letter exceeds number of options
                if answer_letter not in options:
                    issues.append(f"mc_answer_letter_out_of_range:{answer_letter}_options:{''.join(options)}")
            else:
                issues.append(f"mc_answer_not_a_letter:{repr(raw[:40])}")

    # 11. Very short non-MC assistant response
    elif assistant_content and len(assistant_content) <= 2:
        issues.append(f"short_non_mc_assistant:{repr(assistant_content)}")

    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from collections import defaultdict
    issue_counts = defaultdict(int)
    # Store up to 20 examples per issue type
    issue_examples = defaultdict(list)
    MAX_EXAMPLES = 20

    total = 0
    flagged = 0

    with open(args.input, "rb") as fin:
        for i, item in enumerate(ijson.items(fin, "data.item")):
            total += 1
            issues = check_item(item)
            if issues:
                flagged += 1
                for iss in issues:
                    # Normalise parameterised keys for counting
                    key = iss.split(":")[0]
                    issue_counts[key] += 1
                    if len(issue_examples[iss]) < MAX_EXAMPLES:
                        issue_examples[iss].append((i, item))
            if total % 500000 == 0:
                print(f"  ...scanned {total:,}, flagged {flagged:,}", flush=True)

    with open(args.output, "w") as out:
        out.write(f"Total items scanned : {total:,}\n")
        out.write(f"Items with >=1 issue: {flagged:,}\n\n")
        out.write("=== Issue counts ===\n")
        for key, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
            out.write(f"  {key:<45} {cnt:,}\n")
        out.write("\n=== Examples (up to 20 per issue) ===\n")
        for iss, examples in sorted(issue_examples.items()):
            out.write(f"\n--- {iss} ({len(examples)} shown) ---\n")
            for idx, item in examples:
                out.write(f"[{idx}] {json.dumps(item, cls=DecimalEncoder)}\n")

    print(f"\nDone. Report saved to {args.output}")
    print(f"Total: {total:,} | Flagged: {flagged:,}")
    print("\nIssue summary:")
    for key, cnt in sorted(issue_counts.items(), key=lambda x: -x[1]):
        print(f"  {key:<45} {cnt:,}")


if __name__ == "__main__":
    main()
