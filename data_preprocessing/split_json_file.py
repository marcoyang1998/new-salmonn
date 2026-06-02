#!/usr/bin/env python3

import argparse
import copy
import json
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a JSON file with structure {'data': [...]} into contiguous shards."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for shard JSON files.")
    parser.add_argument("--num-shards", type=int, required=True, help="Number of shards to create.")
    parser.add_argument("--prefix", type=str, default="part", help="Shard filename prefix.")
    return parser.parse_args()


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        content = json.load(f)

    if not isinstance(content, dict):
        raise ValueError(f"{path} must be a JSON object.")
    if "data" not in content:
        raise ValueError(f"{path} must contain key 'data'.")
    if not isinstance(content["data"], list):
        raise ValueError(f"{path}['data'] must be a list.")
    return content


def save_json(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_items(items: List[Any], num_shards: int) -> List[tuple[int, int]]:
    total = len(items)
    return [
        (shard_idx * total // num_shards, (shard_idx + 1) * total // num_shards)
        for shard_idx in range(num_shards)
    ]


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive.")

    dataset = load_json(args.input)
    items = dataset["data"]
    os.makedirs(args.output_dir, exist_ok=True)

    for shard_idx, (start, end) in enumerate(split_items(items, args.num_shards)):
        shard = copy.deepcopy(dataset)
        shard["data"] = items[start:end]
        shard_path = os.path.join(args.output_dir, f"{args.prefix}_{shard_idx:02d}.json")
        save_json(shard_path, shard)
        print(f"Wrote {shard_path}: [{start}, {end}) ({end - start} samples)")


if __name__ == "__main__":
    main()