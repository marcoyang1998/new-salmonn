#!/usr/bin/env python3

import argparse
from collections import Counter
import json
import os
import random
import re
from typing import Any, Dict, List, Tuple


LIBRISPEECH_OLD_PREFIX = "/mnt/bn/audio-visual-llm-data/yuwenyi/data/librispeech/LibriSpeech"
LIBRISPEECH_NEW_PREFIX = "/mnt/shared-storage-user/brainllm-share/data/LibriSpeech"
AUDIOCAPS_OLD_PREFIX = "/mnt/bn/audio-visual-llm-data/yuwenyi/audiocaps/audiocaps/"
AUDIOCAPS_NEW_PREFIX = "/mnt/shared-storage-user/brainllm-share/data/AudioCaps/audio/"
MILLIONSONG_OLD_PREFIX = "/mnt/bn/audio-visual-llm-data/datasets/MillionSongDatasetSpotify/"
MILLIONSONG_NEW_PREFIX = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/MillionSongDatasetSpotify/"
MUSICNET_OLD_PREFIX = "/mnt/bn/audio-visual-llm-data/datasets/MusicNet/Preprocess/train/"
MUSICNET_NEW_PREFIX = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/musicnet/train_data/"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Gemini multiple-choice QA data into SALMONN processed format."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="/mnt/shared-storage-gpfs2/speechllm-share/yuwenyi/salmonn_data_v1.1/gemini_qa_mc.json",
        help="Path to gemini_qa_mc.json",
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default="salmonn_data_v1.1/GeminiQaMc_train_processed.json",
        help="Output path for train split JSON.",
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default="salmonn_data_v1.1/GeminiQaMc_val_processed.json",
        help="Output path for validation split JSON.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path when --no-split is used (all data in one file).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        default=False,
        help="Write all converted data to a single file instead of splitting into train/val.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="Validation ratio. Default: 0.05",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split.",
    )
    parser.add_argument(
        "--rewrite-librispeech-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rewrite old LibriSpeech path prefix to shared-storage prefix.",
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require fully valid JSON input. Use --no-strict to enable best-effort recovery.",
    )
    return parser.parse_args()


def change_path_librispeech(path: str) -> str:
    path = path.replace(LIBRISPEECH_OLD_PREFIX, LIBRISPEECH_NEW_PREFIX)
    path = path.replace(".wav", ".flac")
    return path


def change_path_audiocaps(path: str) -> str:
    path = path.replace(AUDIOCAPS_OLD_PREFIX, AUDIOCAPS_NEW_PREFIX)
    dir_name = os.path.dirname(path)
    filename = os.path.basename(path)
    filename = "Y" + filename
    return os.path.join(dir_name, filename)


def change_path_wavcaps(path: str) -> str:
    filename = os.path.basename(path)
    subset = path.split("/")[-2]
    new_path = f"/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/WavCaps/Zip_files/{subset}/audio/{filename}"
    new_path = new_path.replace(".wav", ".flac")
    return new_path


def change_path_millionsong(path: str) -> str:
    return path.replace(MILLIONSONG_OLD_PREFIX, MILLIONSONG_NEW_PREFIX)

def change_path_musicnet(path: str) -> str:
    return path.replace(MUSICNET_OLD_PREFIX, MUSICNET_NEW_PREFIX)

def rewrite_audio_path(path: str, rewrite_librispeech_path: bool) -> str:
    if not rewrite_librispeech_path:
        return path

    audio_lower = path.lower()
    if "librispeech" in audio_lower:
        return change_path_librispeech(path)
    if "audiocaps" in audio_lower:
        return change_path_audiocaps(path)
    if "wavcaps" in audio_lower:
        return change_path_wavcaps(path)
    if "millionsong" in audio_lower:
        return change_path_millionsong(path)
    if "musicnet" in audio_lower:
        return change_path_musicnet(path)
    return path


def infer_dataset_name(path: str) -> str:
    audio_lower = path.lower()
    if "librispeech" in audio_lower:
        return "librispeech"
    if "audiocaps" in audio_lower:
        return "audiocaps"
    if "wavcaps" in audio_lower:
        return "wavcaps"
    if "millionsong" in audio_lower:
        return "millionsong"
    if "musicnet" in audio_lower:
        return "musicnet"
    return "other"


def normalize_answer_label(answer: str) -> str:
    answer = answer.strip()
    if answer.lower().startswith("option "):
        answer = answer.split()[-1]
    answer = answer.upper()
    if len(answer) != 1 or not ("A" <= answer <= "Z"):
        raise ValueError(f"Unsupported answer format: {answer}")
    return answer


def build_question_text(question: str, options: List[Tuple[str, str]]) -> str:
    option_lines = [f"Option {label}: {content}" for label, content in options]
    return (
        "<audio>Answer the following multiple-choice question using only the correct option.\n"
        f"Question: {question}\n"
        "Choices:\n"
        + "\n".join(option_lines)
        + "\nPlease output your final answer with a single letter. "
        "For example, if you think the answer is Option A, please just output 'A'"
    )

def build_question_text_v0(question: str, options: List[Tuple[str, str]]) -> str:
    option_lines = [f"Option {label}: {content}" for label, content in options]
    return (
        "<audio>Answer the following multiple-choice question using only the correct option.\n"
        f"Question: {question}\n"
        "Choices:\n"
        + "\n".join(option_lines)
        + '\nPlease output your final answer with a single letter. For example, if you think the answer is Option B, please just output \'B\''
    )


def convert_entry(entry: Dict[str, Any], rewrite_librispeech_path: bool) -> Dict[str, Any]:
    path = rewrite_audio_path(entry["path"], rewrite_librispeech_path)
    qa = entry["QA"]
    question = qa["question"].strip()

    options: List[Tuple[str, str]] = []
    for key, value in qa.items():
        key_lower = key.lower()
        if key_lower.startswith("option "):
            label = key.split()[-1].upper()
            options.append((label, str(value).strip()))

    if not options:
        raise ValueError(f"No options found in entry: {entry}")

    options = sorted(options, key=lambda x: x[0])
    answer_label = normalize_answer_label(qa["answer"])
    valid_labels = {label for label, _ in options}
    if answer_label not in valid_labels:
        raise ValueError(f"Answer label {answer_label} not in options {sorted(valid_labels)}")

    user_text = build_question_text(question, options)
    assistant_text = answer_label

    mc_options = []
    for label, content in options:
        mc_options.append(
            {
                "label": label,
                "content": content,
                "is_correct": label == answer_label,
            }
        )

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "audios": [path],
        "task_type": "qa_mc",
        "mc_question": question,
        "mc_options": mc_options,
        "mc_answer_label": answer_label,
    }


def split_train_val(samples: List[Dict[str, Any]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    val_size = max(1, int(round(len(samples) * val_ratio)))
    val_indices = set(indices[:val_size])
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_indices]
    val_samples = [samples[i] for i in range(len(samples)) if i in val_indices]
    return train_samples, val_samples


def save_json(path: str, data: Dict[str, Any]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_records_with_fallback(path: str, strict: bool = True) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    try:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("Expected input to be a JSON list")
        return data
    except json.JSONDecodeError as err:
        if strict:
            raise ValueError(
                f"Input is not valid JSON ({err}). If you want best-effort recovery, rerun with --no-strict."
            ) from err
        print(f"Warning: input JSON is malformed ({err}). Falling back to best-effort recovery.")

    
    stripped = content.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        raise ValueError("Input must be a JSON array.")
    body = stripped[1:-1].strip()
    if not body:
        return []

    chunks: List[str] = []
    depth = 0
    start_idx = -1
    in_string = False
    escape = False
    for i, ch in enumerate(body):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue

        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start_idx != -1:
                    chunks.append(body[start_idx : i + 1])
                    start_idx = -1

    recovered: List[Dict[str, Any]] = []
    failed = 0
    for candidate in chunks:
        try:
            item = json.loads(candidate)
            if isinstance(item, dict):
                recovered.append(item)
            else:
                failed += 1
        except json.JSONDecodeError:
            failed += 1

    if depth != 0:
        failed += 1

    print(f"Recovered records: {len(recovered)}, skipped malformed records: {failed}")
    return recovered


def main() -> None:
    args = parse_args()

    raw_data = load_records_with_fallback(args.input, strict=args.strict)

    converted: List[Dict[str, Any]] = []
    skipped_missing_audio = 0
    skipped_by_dataset: Counter[str] = Counter()
    for idx, item in enumerate(raw_data):
        converted_entry = convert_entry(item, args.rewrite_librispeech_path)

        dataset_name = infer_dataset_name(str(item.get("path", "")))
        missing_paths = [audio for audio in converted_entry.get("audios", []) if not os.path.exists(audio)]
        if missing_paths:
            skipped_missing_audio += 1
            skipped_by_dataset[dataset_name] += 1
            print(
                f"Skipping entry {idx} ({dataset_name}): missing audio file(s): "
                + ", ".join(missing_paths)
            )
            continue

        converted.append(converted_entry)

    if args.no_split:
        output_path = args.output or args.train_output
        save_json(output_path, {"data": converted})
        print(f"Converted total: {len(converted)}")
        print(f"Skipped (missing audio): {skipped_missing_audio}")
        if skipped_by_dataset:
            print("Skipped by dataset (missing audio):")
            for dataset_name, count in sorted(skipped_by_dataset.items()):
                print(f"  {dataset_name}: {count}")
        print(f"All samples: {len(converted)} -> {output_path}")
        return

    train_samples, val_samples = split_train_val(converted, args.val_ratio, args.seed)

    save_json(args.train_output, {"data": train_samples})
    save_json(args.val_output, {"data": val_samples})

    print(f"Converted total: {len(converted)}")
    print(f"Skipped (missing audio): {skipped_missing_audio}")
    if skipped_by_dataset:
        print("Skipped by dataset (missing audio):")
        for dataset_name, count in sorted(skipped_by_dataset.items()):
            print(f"  {dataset_name}: {count}")
    print(f"Train samples: {len(train_samples)} -> {args.train_output}")
    print(f"Val samples:   {len(val_samples)} -> {args.val_output}")


if __name__ == "__main__":
    main()