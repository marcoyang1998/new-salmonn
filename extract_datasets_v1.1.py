#!/usr/bin/env python3
"""Extract dataset-specific entries from a JSON file with v1.1 schema.

Expected input format:
    {
      "data": [
        {
          "messages": [...],
          "audios": ["/path/to/audio1.wav", ...]
        },
        ...
      ]
    }

Example:
    python extract_datasets_v1.1.py \
        --input /path/to/input.json \
        --datasets librispeech gigaspeech audiocaps \
        --output ./salmonn_data/extracted/librispeech_gigaspeech_audiocaps_v1_1.json
"""

import argparse
import json
import os
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional


DATASETS = ["librispeech", "gigaspeech", "audiocaps", "iemocap", "voxceleb1", "librimix", "clotho", "commonvoice", "wavcaps"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract entries by matching dataset names in each item of the 'audios' field."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        help="Dataset names to extract (matched as substrings in audio paths).",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_entries(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        for item in data["data"]:
            if isinstance(item, dict):
                yield item


def change_path_librispeech(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/yuwenyi/data/librispeech/LibriSpeech",
        "/mnt/shared-storage-user/brainllm-share/data/LibriSpeech",
    )
    path = path.replace(".wav", ".flac")
    return path


def change_path_gigaspeech(path: str) -> str:
    path = path.split("/extracted/")[-1]
    path = path.split("/", 1)[-1]
    subset = path.split("/")[0].split("_")[0]
    if subset == "xs":
        path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/xs_files/" + path
    elif subset == "s":
        path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/s_files_additional/" + path
    elif subset == "m":
        path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/m_files_additional/" + path
    else:
        raise ValueError(f"Unexpected GigaSpeech subset in path: {path}")
    return path


def change_path_audiocaps(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/yuwenyi/audiocaps/audiocaps/",
        "/mnt/shared-storage-user/brainllm-share/data/AudioCaps/audio/",
    )
    dir_name = os.path.dirname(path)
    filename = os.path.basename(path)
    filename = "Y" + filename
    return os.path.join(dir_name, filename)

def change_path_iemocap(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/iemocap/IEMOCAP_full_release/",
        "/mnt/shared-storage-user/brainllm-share/data/IEMOCAP/",
    )
    return path

def change_path_voxceleb(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/",
        "/mnt/shared-storage-user/brainllm-share/data/Voxceleb1/",
    )
    return path

def change_path_librimix(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/LibriMix/mixdata/Libri2Mix/",
        "/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri2Mix/"
    )
    return path

def change_path_clotho(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/clotho/Preprocess/train/",
        "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/clotho/development/",
    )
    return path

def change_commonvoice(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/yuwenyi/commonvoice/train/",
        "/mnt/shared-storage-user/brainllm-share/xiaoyu/superb_data/covost2/en/clips/",
    )
    return path

def change_wavcaps(path: str) -> str:
    filename = os.path.basename(path)
    subset = path.split("/")[-2]
    new_path = f"/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/WavCaps/Zip_files/{subset}/audio/{filename}"
    new_path = new_path.replace(".wav", ".flac")
    return new_path

def detect_dataset_name(audio_path: str, wanted: List[str]) -> Optional[str]:
    audio_lower = audio_path.lower()
    for name in wanted:
        if name in audio_lower:
            return name
    return None


def rewrite_audio_path(name: str, audio_path: str) -> str:
    if name == "librispeech":
        new_path = change_path_librispeech(audio_path)
        assert os.path.exists(new_path), f"LibriSpeech path does not exist: {new_path}"
        return new_path

    elif name == "gigaspeech":
        new_path = change_path_gigaspeech(audio_path)
        assert os.path.exists(new_path), f"GigaSpeech path does not exist: {new_path}"
        return new_path

    elif name == "audiocaps":
        new_path = change_path_audiocaps(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: AudioCaps path does not exist: {new_path}")
        return new_path
    
    elif name == "iemocap":
        new_path = change_path_iemocap(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: IEMOCAP path does not exist: {new_path}")
        return new_path
    
    elif name == "voxceleb1":
        new_path = change_path_voxceleb(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: Voxceleb path does not exist: {new_path}")
        return new_path
    
    elif name == "librimix":
        new_path = change_path_librimix(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: LibriMix path does not exist: {new_path}")
        return new_path
    
    elif name == "clotho":
        new_path = change_path_clotho(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: Clotho path does not exist: {new_path}")
        return new_path
    
    elif name == "commonvoice":
        new_path = change_commonvoice(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: CommonVoice path does not exist: {new_path}")
        return new_path
    elif name == "wavcaps":
        new_path = change_wavcaps(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: WavCaps path does not exist: {new_path}")
        return new_path


def main() -> None:
    args = parse_args()
    import pdb; pdb.set_trace()
    data = load_json(args.input)

    wanted = [name.strip().lower() for name in args.datasets]
    extracted: Dict[str, List[Dict[str, Any]]] = {name: [] for name in wanted}
    combined: List[Dict[str, Any]] = []

    for entry in tqdm(iter_entries(data)):
        audios = entry.get("audios")
        if not isinstance(audios, list):
            continue

        matched_names = set()
        updated_audios: List[Any] = []

        ALL_AUDIOS_VALID = True
        for audio in audios:
            if not isinstance(audio, str):
                updated_audios.append(audio)
                continue

            matched_name = detect_dataset_name(audio, wanted)
            if matched_name is None:
                updated_audios.append(audio)
                continue

            updated_audio = rewrite_audio_path(matched_name, audio)
            if not os.path.exists(updated_audio):
                ALL_AUDIOS_VALID = False
                break
            matched_names.add(matched_name)
            updated_audios.append(updated_audio)

        if not matched_names or not ALL_AUDIOS_VALID:
            continue

        new_entry = dict(entry)
        new_entry["audios"] = updated_audios

        for name in matched_names:
            extracted[name].append(new_entry)
        combined.append(new_entry)

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"data": combined}, f, ensure_ascii=True, indent=2)

    total = sum(len(v) for v in extracted.values())
    print(f"Original dataset has {len(data['data'])} entries, wrote {len(combined)} entries (dataset matches counted: {total}) to {args.output}")


if __name__ == "__main__":
    main()
