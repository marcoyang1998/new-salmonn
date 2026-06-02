#!/usr/bin/env python3
import argparse
import csv
import json
import wave
from pathlib import Path


DEFAULT_PROMPT = "<audio>Identify the primary sound event in this audio clip."
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a SALMONN-format JSON manifest for ESC-50."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/ESC_50"),
        help="Path to the ESC-50 dataset root.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=REPO_ROOT / "data" / "esc50_salmonn.json",
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="User prompt stored in each SALMONN sample.",
    )
    return parser.parse_args()


def load_label_map(label_csv: Path) -> dict[int, str]:
    label_map: dict[int, str] = {}
    with label_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            label_map[int(row["index"])] = row["display_name"]
    return label_map


def get_duration_seconds(audio_path: Path) -> float:
    with wave.open(str(audio_path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def build_entry(row: dict[str, str], audio_root: Path, label_map: dict[int, str], prompt: str) -> dict:
    filename = row["filename"]
    audio_path = audio_root / filename
    if not audio_path.exists():
        raise FileNotFoundError(f"Missing audio file: {audio_path}")

    target = int(row["target"])
    label = label_map.get(target, row["category"])
    duration = get_duration_seconds(audio_path)

    return {
        "messages": [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": label,
            },
        ],
        "audios": [str(audio_path.resolve())],
        "durations": [duration],
        "task_type": "sound_event_classification",
        "dataset": "ESC-50",
        "filename": filename,
        "target": target,
        "category": row["category"],
        "label": label,
        "fold": int(row["fold"]),
        "esc10": row["esc10"].strip().lower() == "true",
        "src_file": row["src_file"],
        "take": row["take"],
    }


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    meta_csv = dataset_root / "meta" / "esc50.csv"
    label_csv = dataset_root / "esc_class_labels_indices.csv"
    audio_root = dataset_root / "audio"

    if not meta_csv.exists():
        raise FileNotFoundError(f"Missing metadata CSV: {meta_csv}")
    if not label_csv.exists():
        raise FileNotFoundError(f"Missing label CSV: {label_csv}")
    if not audio_root.exists():
        raise FileNotFoundError(f"Missing audio directory: {audio_root}")

    label_map = load_label_map(label_csv)

    data = []
    with meta_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            data.append(build_entry(row, audio_root, label_map, args.prompt))

    output_path = args.output_json
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "dataset": "ESC-50",
        "audio_root": str(audio_root),
        "num_classes": len(label_map),
        "num_samples": len(data),
        "label_map": {str(index): label for index, label in sorted(label_map.items())},
        "data": data,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(f"Wrote {len(data)} samples to {output_path}")


if __name__ == "__main__":
    main()