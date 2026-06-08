import argparse
import json
import re
from collections import defaultdict


def load_records(path):
    """Load MMSU results from either JSONL or a JSON array file."""
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)

        if first_char == "[":
            return json.load(f)

        records = []
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}: {exc}") from exc
        return records


def extract_choice(response):
    """Extract an A/B/C/D choice from common MMSU response formats."""
    if response is None:
        return None

    text = str(response).strip().replace("\n", "")
    if not text or text == "None":
        return None

    upper_text = text.upper()
    if upper_text[0] in {"A", "B", "C", "D"}:
        return upper_text[0]

    match = re.search(r"\b([ABCD])\b", upper_text)
    if match:
        return match.group(1)

    return None


def is_correct(record):
    prediction = extract_choice(record.get("response"))
    if prediction is None:
        return False

    answer_gt = str(record.get("answer_gt", "")).strip()
    choices = {
        "A": record.get("choice_a", ""),
        "B": record.get("choice_b", ""),
        "C": record.get("choice_c", ""),
        "D": record.get("choice_d", ""),
    }

    return str(choices.get(prediction, "")).strip() == answer_gt


def calculate_task_accuracy(records):
    task_counts = defaultdict(lambda: {"correct": 0, "total": 0})

    for record in records:
        task_name = record.get("task_name") or "UNKNOWN_TASK"
        task_counts[task_name]["total"] += 1
        if is_correct(record):
            task_counts[task_name]["correct"] += 1

    return task_counts


def print_task_accuracy(task_counts):
    overall_correct = 0
    overall_total = 0

    for task_name in sorted(task_counts):
        correct = task_counts[task_name]["correct"]
        total = task_counts[task_name]["total"]
        accuracy = correct / total if total else 0.0
        overall_correct += correct
        overall_total += total
        print(f"{task_name}: {correct}/{total} ({accuracy:.4f})")

    overall_accuracy = overall_correct / overall_total if overall_total else 0.0
    print(f"Overall: {overall_correct}/{overall_total} ({overall_accuracy:.4f})")


def main():
    parser = argparse.ArgumentParser(
        description="Print task_name-level accuracy for MMSU result files."
    )
    parser.add_argument("result_path", help="Path to an MMSU result JSONL/JSON file")
    args = parser.parse_args()

    records = load_records(args.result_path)
    task_counts = calculate_task_accuracy(records)
    print_task_accuracy(task_counts)


if __name__ == "__main__":
    main()
