#!/usr/bin/env python3
"""Analyze option-position bias in MCQ result files.

Supports both:
  * MMAR-style JSON list with "choices", "answer", "model_prediction"
  * MMSU-style JSONL/JSON with "choice_a"..."choice_h", "answer_gt", "response"

Reports:
  1. Overall predicted option proportions.
  2. Predicted A/B proportions on binary questions.
  3. Predicted A/B/C/D proportions on 4-way questions.

Each section also reports reliability for each predicted option, i.e.
accuracy conditioned on the model predicting that option.
"""

from __future__ import annotations

import argparse
import json
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


LETTERS = string.ascii_uppercase


def load_records(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Unsupported JSON structure in {path}")


def normalize_text(value: Any) -> str:
    return str(value).strip()


def get_choices(record: Dict[str, Any]) -> List[str]:
    if isinstance(record.get("choices"), list):
        return [normalize_text(choice) for choice in record["choices"]]

    choices = []
    for label in LETTERS[:8]:
        key = f"choice_{label.lower()}"
        if key in record:
            choices.append(normalize_text(record[key]))
    return choices


def index_from_letter(value: Any, num_choices: int) -> Optional[int]:
    text = normalize_text(value).strip(".):：").upper()
    if len(text) == 1 and text in LETTERS[:num_choices]:
        return LETTERS.index(text)
    if text.startswith("OPTION "):
        suffix = text.replace("OPTION ", "", 1).strip(".):：")
        if len(suffix) == 1 and suffix in LETTERS[:num_choices]:
            return LETTERS.index(suffix)
    return None


def index_from_text(value: Any, choices: List[str]) -> Optional[int]:
    text = normalize_text(value)
    if text in choices:
        return choices.index(text)

    # Some result files keep harmless trailing whitespace in choices.
    normalized_choices = [choice.strip() for choice in choices]
    if text.strip() in normalized_choices:
        return normalized_choices.index(text.strip())
    return None


def prediction_index(record: Dict[str, Any], choices: List[str]) -> Optional[int]:
    for key in ("response", "prediction", "predicted_label"):
        if key in record:
            index = index_from_letter(record[key], len(choices))
            if index is not None:
                return index

    for key in ("model_prediction", "predicted_answer"):
        if key in record:
            index = index_from_letter(record[key], len(choices))
            if index is not None:
                return index
            index = index_from_text(record[key], choices)
            if index is not None:
                return index
    return None


def gold_index(record: Dict[str, Any], choices: List[str]) -> Optional[int]:
    for key in ("answer_label", "gold_label"):
        if key in record:
            index = index_from_letter(record[key], len(choices))
            if index is not None:
                return index

    for key in ("answer", "answer_gt", "gold_answer"):
        if key in record:
            index = index_from_letter(record[key], len(choices))
            if index is not None:
                return index
            index = index_from_text(record[key], choices)
            if index is not None:
                return index
    return None


def collect_examples(records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    examples = []
    unmatched_predictions = 0
    unmatched_gold = 0

    for record in records:
        choices = get_choices(record)
        if not choices:
            continue

        pred_idx = prediction_index(record, choices)
        gold_idx = gold_index(record, choices)
        if pred_idx is None:
            unmatched_predictions += 1
        if gold_idx is None:
            unmatched_gold += 1

        examples.append(
            {
                "record": record,
                "choices": choices,
                "num_choices": len(choices),
                "pred_idx": pred_idx,
                "gold_idx": gold_idx,
                "correct": pred_idx is not None and gold_idx is not None and pred_idx == gold_idx,
            }
        )

    return examples, unmatched_predictions, unmatched_gold


def percent(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.00%"
    return f"{numerator / denominator * 100:.2f}%"


def section_stats(examples: List[Dict[str, Any]], labels: str) -> Dict[str, Dict[str, int]]:
    stats = {label: {"predicted": 0, "with_gold": 0, "correct": 0} for label in labels}
    for example in examples:
        pred_idx = example["pred_idx"]
        if pred_idx is None or pred_idx >= len(labels):
            continue
        label = labels[pred_idx]
        stats[label]["predicted"] += 1
        if example["gold_idx"] is not None:
            stats[label]["with_gold"] += 1
            stats[label]["correct"] += int(example["correct"])
    return stats


def print_section(title: str, examples: List[Dict[str, Any]], labels: str) -> None:
    stats = section_stats(examples, labels)
    total_matched = sum(item["predicted"] for item in stats.values())
    total_with_gold = sum(item["with_gold"] for item in stats.values())
    print(f"\n{title}")
    print(f"Examples             : {len(examples):,}")
    print(f"Matched predictions  : {total_matched:,}")
    print(f"Gold labels available: {total_with_gold:,}")
    print("Option  Correct/Predicted  Proportion  Reliability")
    print("------  -----------------  ----------  -----------")
    for label in labels:
        predicted = stats[label]["predicted"]
        with_gold = stats[label]["with_gold"]
        correct = stats[label]["correct"]
        reliability = percent(correct, with_gold) if with_gold else "N/A"
        print(
            f"{label:<6}  "
            f"{correct:>5}/{predicted:<8}  "
            f"{percent(predicted, total_matched):>10}  "
            f"{reliability:>11}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Result JSON or JSONL file to analyze.")
    args = parser.parse_args()

    records = load_records(args.path)
    examples, unmatched_predictions, unmatched_gold = collect_examples(records)
    matched = [example for example in examples if example["pred_idx"] is not None]
    correct = sum(example["correct"] for example in examples)

    choice_lengths = Counter(example["num_choices"] for example in examples)
    gold_counts = Counter(example["gold_idx"] for example in examples if example["gold_idx"] is not None)
    gold_available = sum(1 for example in examples if example["gold_idx"] is not None)

    print(f"File                 : {args.path}")
    print(f"Examples             : {len(examples):,}")
    print(f"Choice lengths       : {dict(sorted(choice_lengths.items()))}")
    print(f"Unmatched prediction : {unmatched_predictions:,}")
    print(f"Gold labels available: {gold_available:,}")
    print(f"Missing/unmatched gold: {unmatched_gold:,}")
    if gold_available:
        print(f"Overall accuracy     : {correct:,}/{gold_available:,} ({percent(correct, gold_available)})")
        print(
            "Gold option counts   : "
            + ", ".join(
                f"{LETTERS[index]}={count:,}" for index, count in sorted(gold_counts.items())
            )
        )
    else:
        print("Overall accuracy     : N/A (no ground-truth answers found)")
        print("Gold option counts   : N/A")

    print_section("Overall Predicted A/B/C/D", matched, "ABCD")

    binary = [example for example in matched if example["num_choices"] == 2]
    print_section("Binary Questions Predicted A/B", binary, "AB")

    four_way = [example for example in matched if example["num_choices"] == 4]
    print_section("4-Way Questions Predicted A/B/C/D", four_way, "ABCD")


if __name__ == "__main__":
    main()
