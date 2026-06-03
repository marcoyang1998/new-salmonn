#!/usr/bin/env python3

import argparse
import json
import os
import random
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample items from a JSON dataset while preserving the original top-level format."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSON file.")
    parser.add_argument(
        "--num-samples",
        type=int,
        required=True,
        help="Number of items to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def detect_top_level_key(content: Dict[str, Any]) -> str:
    if "data" in content and isinstance(content["data"], list):
        return "data"
    if "annotation" in content and isinstance(content["annotation"], list):
        return "annotation"
    raise ValueError("Input JSON must contain a top-level list under 'data' or 'annotation'.")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if not isinstance(content, dict):
        raise ValueError(f"{path} must be a JSON object.")
    return content


def save_json(path: str, content: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(content, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    if args.num_samples < 0:
        raise ValueError("--num-samples must be >= 0.")

    data_obj = load_json(args.input)
    list_key = detect_top_level_key(data_obj)
    items: List[Any] = data_obj[list_key]

    if args.num_samples > len(items):
        raise ValueError(
            f"Requested {args.num_samples} samples, but only {len(items)} items are available."
        )

    rng = random.Random(args.seed)
    sampled_items = rng.sample(items, args.num_samples)

    output_obj = dict(data_obj)
    output_obj[list_key] = sampled_items
    save_json(args.output, output_obj)

    print(
        json.dumps(
            {
                "input": args.input,
                "output": args.output,
                "top_level_key": list_key,
                "total_items": len(items),
                "sampled_items": len(sampled_items),
                "seed": args.seed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
