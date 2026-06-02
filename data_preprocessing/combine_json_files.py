#!/usr/bin/env python3

import argparse
import json
import os
import warnings
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine multiple JSON files with structure {'data': [...]} into one."
    )
    parser.add_argument(
        "--inputs",
        type=str,
        nargs="+",
        required=True,
        help="One or more input JSON files to combine.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to the output combined JSON file.",
    )
    return parser.parse_args() 


def load_json_data(path: str) -> List[Any]:
    with open(path, "r", encoding="utf-8") as f:
        content: Dict[str, Any] = json.load(f)

    if not isinstance(content, dict):
        raise ValueError(f"{path} must be a JSON object.")
    if "data" not in content:
        raise ValueError(f"{path} must contain key 'data'.")
    if not isinstance(content["data"], list):
        raise ValueError(f"{path}['data'] must be a list.")

    return content["data"]


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()

    combined_data: List[Any] = []
    item_counts: List[int] = []
    for input_path in args.inputs:
        print(f"Reading {input_path}...")
        input_data = load_json_data(input_path)

        if len(input_data) > 0:
            first_item = input_data[0]
            if not isinstance(first_item, dict) or "task_type" not in first_item:
                warnings.warn(
                    f"{input_path}: first item in 'data' does not contain field 'task_type'.",
                    stacklevel=1,
                )

        item_counts.append(len(input_data))
        combined_data.extend(input_data)

    combined = {"data": combined_data}
    save_json(args.output, combined)

    print(
        f"Saved combined JSON to {args.output}. "
        f"items: {' + '.join(str(x) for x in item_counts)} = {len(combined['data'])}"
    )


if __name__ == "__main__":
    main()
