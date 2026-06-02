#!/usr/bin/env python3

import argparse
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split stage2 dataset by prompt groups and assign task_type from <TASK> tags in stage2_prompts.txt."
    )
    parser.add_argument(
        "--input-json",
        default="salmonn_data_v1.1/stage2/stage2_train_with_durations.json",
        help="Input dataset JSON (expects {'data': [...]})",
    )
    parser.add_argument(
        "--prompts-file",
        default="salmonn_data_v1.1/prompts/stage2_prompts.txt",
        help="Prompt definition file with group headers and optional <TASK> tag lines.",
    )
    parser.add_argument(
        "--output-dir",
        default="salmonn_data_v1.1/task_splits_by_prompt",
        help="Directory to save split JSON files (by group name).",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent for output manifests.",
    )
    parser.add_argument(
        "--clear-output-dir",
        action="store_true",
        help="Remove output directory first if it already exists.",
    )
    return parser.parse_args()


def sanitize_name(name: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return safe or "group"


def parse_prompts(prompts_path: Path) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    section_to_prompts: Dict[str, Set[str]] = defaultdict(set)
    section_to_task_type: Dict[str, str] = {}

    current_section: Optional[str] = None
    content_re = re.compile(r'^"content"\s*:\s*"(.*)"\s*$')
    task_re = re.compile(r"^<([^<>]+)>$")

    with prompts_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if line.endswith(":"):
                current_section = line[:-1].strip()
                continue

            if current_section is None:
                continue

            task_match = task_re.match(line)
            if task_match:
                section_to_task_type[current_section] = task_match.group(1).strip()
                continue

            content_match = content_re.match(line)
            if content_match:
                section_to_prompts[current_section].add(content_match.group(1))

    return section_to_prompts, section_to_task_type


def get_user_prompt(item: dict) -> str:
    for msg in item.get("messages", []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", "")).strip()
    return ""


def clone_with_task_type(item: dict, task_type: str) -> dict:
    out = dict(item)
    out["task_type"] = task_type
    return out


def main() -> None:
    args = parse_args()

    input_json = Path(args.input_json)
    prompts_file = Path(args.prompts_file)
    output_dir = Path(args.output_dir)

    section_to_prompts, section_to_task_type = parse_prompts(prompts_file)

    prompt_to_section: Dict[str, str] = {}
    for section, prompts in section_to_prompts.items():
        for prompt in prompts:
            prompt_to_section[prompt] = section

    with input_json.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Input JSON must be a dict with key 'data' as list.")

    items: List[dict] = payload["data"]

    if args.clear_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    buckets: Dict[str, List[dict]] = defaultdict(list)
    unmatched_prompts: Dict[str, int] = defaultdict(int)

    for item in items:
        if not isinstance(item, dict):
            continue

        user_prompt = get_user_prompt(item)
        section = prompt_to_section.get(user_prompt)

        if section is None:
            group_name = "QA"
            task_type = "QA"
            unmatched_prompts[user_prompt] += 1
        else:
            group_name = section
            task_type = section_to_task_type.get(section, section)

        buckets[group_name].append(clone_with_task_type(item, task_type))

    summary: List[Tuple[str, int, str, str]] = []
    written_total = 0

    for group_name in sorted(buckets.keys()):
        safe_name = sanitize_name(group_name)
        out_path = output_dir / f"{safe_name}.json"
        out_payload = dict(payload)
        out_payload["data"] = buckets[group_name]

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, ensure_ascii=False, indent=args.indent)
            f.write("\n")

        count = len(buckets[group_name])
        written_total += count

        group_task_types = sorted({entry.get("task_type", "") for entry in buckets[group_name]})
        task_types_str = ",".join([t for t in group_task_types if t]) or "<EMPTY>"
        summary.append((group_name, count, out_path.as_posix(), task_types_str))

    print(f"input_total={len(items)}")
    print(f"group_files={len(summary)}")
    print(f"written_total={written_total}")
    for group_name, count, out_path, task_types in summary:
        print(f"{group_name}\t{count}\t{task_types}\t{out_path}")

    if unmatched_prompts:
        print("---UNMATCHED_TOP10---")
        top = sorted(unmatched_prompts.items(), key=lambda x: x[1], reverse=True)[:10]
        for prompt, count in top:
            shown = (prompt or "<EMPTY>").replace("\t", " ").replace("\n", " ")
            print(f"{count}\t{shown}")


if __name__ == "__main__":
    main()
