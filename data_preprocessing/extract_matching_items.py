#!/usr/bin/env python3

import argparse
import json
import os
import re
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract SALMONN training items whose selected field matches a regex pattern. "
            "Input JSON must contain top-level key 'data'."
        )
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to output JSON file.",
    )
    parser.add_argument(
        "--field",
        type=str,
        required=True,
        choices=["audios", "task_type"],
        help="Which field to check for a match.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Regex pattern to match against the selected field.",
    )
    parser.add_argument(
        "--ignore_case",
        action="store_true",
        help="Enable case-insensitive regex matching.",
    )
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if not isinstance(content, dict):
        raise ValueError(f"{path} must be a JSON object.")
    if "data" not in content:
        raise ValueError(f"{path} must contain top-level key 'data'.")
    if not isinstance(content["data"], list):
        raise ValueError(f"{path}['data'] must be a list.")

    return content


def item_matches(item: Dict[str, Any], field: str, regex: re.Pattern[str]) -> bool:
    value = item.get(field)

    if field == "task_type":
        if isinstance(value, str):
            return regex.search(value) is not None
        return False

    # field == "audios"
    if isinstance(value, str):
        return regex.search(value) is not None

    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, str):
                if regex.search(entry) is not None:
                    return True
            else:
                if regex.search(str(entry)) is not None:
                    return True
        return False

    return False


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    flags = re.IGNORECASE if args.ignore_case else 0
    regex = re.compile(args.pattern, flags)

    dataset = load_json(args.input_json)
    items: List[Any] = dataset["data"]

    filtered_items: List[Any] = []
    for item in items:
        if isinstance(item, dict) and item_matches(item, args.field, regex):
            filtered_items.append(item)

    output_obj = dict(dataset)
    output_obj["data"] = filtered_items

    save_json(args.output_json, output_obj)

    print(
        json.dumps(
            {
                "input_json": args.input_json,
                "output_json": args.output_json,
                "field": args.field,
                "pattern": args.pattern,
                "ignore_case": args.ignore_case,
                "total_items": len(items),
                "matched_items": len(filtered_items),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
