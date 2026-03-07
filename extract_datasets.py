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
import multiprocessing
import os
import soundfile as sf
from tqdm import tqdm
from functools import partial
from typing import Any, Dict, Iterable, List, Optional, Tuple


DATASETS = [
    "librispeech", "gigaspeech", "audiocaps", "iemocap", "voxceleb1", "librimix", "clotho", "commonvoice", "wavcaps", "musiccaps", "millionsong"
]

audioset_youtube_ids = {}
with open("/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl/data/audioset_manifest/youtube_ids_all.txt") as f:
    data = f.read().strip().split("\n")
    for line in data:
        ytid, audio = line.split()
        audioset_youtube_ids[ytid] = audio

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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=40,
        help="Number of worker processes for multiprocessing.",
    )
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_entries(data: Any, key: str = "data") -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get(key), list):
        for item in data[key]:
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
    if "subset_test" not in path:
        path = path.split("/extracted/")[-1]
        path = path.split("/", 1)[-1]
        subset = path.split("/")[0].split("_")[0]
        if subset == "xs":
            path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/xs_files/" + path
        elif subset == "s":
            path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/s_files_additional/" + path
        elif subset == "m":
            path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/m_files_additional/" + path
        elif subset == "subset_test":
            path = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/test_files/" + path
        else:
            raise ValueError(f"Unexpected GigaSpeech subset in path: {path}")
    else:
        # test set
        wav_file = path.split("/")[-1]
        for chunk in [0,1,2]:
            potential_path = f"/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/test_files/test_chunks_000{chunk}/{wav_file}"
            if os.path.exists(potential_path):
                path = potential_path
                break
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
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/clotho/Preprocess/test/",
        "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/clotho/evaluation/",
    )
    return path

def change_commonvoice(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/yuwenyi/commonvoice/train/",
        "/mnt/shared-storage-user/brainllm-share/xiaoyu/superb_data/covost2/en/clips/",
    )
    path = path.replace(".wav", ".mp3")
    return path

def change_wavcaps(path: str) -> str:
    filename = os.path.basename(path)
    subset = path.split("/")[-2]
    new_path = f"/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/WavCaps/Zip_files/{subset}/audio/{filename}"
    new_path = new_path.replace(".wav", ".flac")
    return new_path

def change_musiccaps(path: str) -> str:
    filename = os.path.basename(path)
    ytid = filename.replace(".wav", "")
    if ytid not in audioset_youtube_ids:
        return ""
    return audioset_youtube_ids[ytid]
    
def change_millionsong(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/MillionSongDatasetSpotify/",
        "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/MillionSongDatasetSpotify/"
    )
    return path

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
    elif name == "musiccaps":
        new_path = change_musiccaps(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: MusicCaps path does not exist: {new_path}")
        return new_path
    elif name == "millionsong":
        new_path = change_millionsong(audio_path)
        if not os.path.exists(new_path):
            print(f"Warning: MillionSongDatasetSpotify path does not exist: {new_path}")
        return new_path


def process_entry(entry: Dict[str, Any], wanted: List[str]) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """Process a single entry."""
    audios = entry.get("audios")
    failed_counts = {name: 0 for name in wanted}
    
    if not isinstance(audios, list):
        return None, failed_counts

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
            failed_counts[matched_name] += 1
            break
        
        if updated_audio.endswith(".mp3"):
            try:
                _ = sf.info(updated_audio)
            except Exception as e:
                ALL_AUDIOS_VALID = False
                failed_counts[matched_name] += 1
                break

        matched_names.add(matched_name)
        updated_audios.append(updated_audio)

    if not matched_names or not ALL_AUDIOS_VALID:
        return None, failed_counts

    new_entry = dict(entry)
    new_entry["audios"] = updated_audios
    # We add matched names to the entry so we don't have to re-detect in main
    new_entry["_matched_names"] = list(matched_names)

    return new_entry, failed_counts


def main() -> None:
    args = parse_args()
    data = load_json(args.input)

    wanted = [name.strip().lower() for name in args.datasets]
    
    entries = list(iter_entries(data))
    worker = partial(process_entry, wanted=wanted)

    extracted: Dict[str, List[Dict[str, Any]]] = {name: [] for name in wanted}
    combined: List[Dict[str, Any]] = []
    failed_total = {name: 0 for name in wanted}

    with multiprocessing.Pool(args.num_workers) as pool:
        results = list(tqdm(pool.imap(worker, entries, chunksize=100), total=len(entries), desc="Processing"))

    for result, fail_counts in results:
        for name, count in fail_counts.items():
            failed_total[name] += count
        
        if result is not None:
            entry = dict(result)
            matched_names = entry.pop("_matched_names")
            combined.append(entry)
            for name in matched_names:
                extracted[name].append(entry)

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({"data": combined}, f, ensure_ascii=True, indent=2)

    total = sum(len(v) for v in extracted.values())
    for name in failed_total:
        print(f"Dataset '{name}': {len(extracted[name])} entries extracted, {failed_total[name]} entries failed due to missing/invalid files.")
    print(f"Original dataset has {len(data['data'])} entries, wrote {len(combined)} entries (dataset matches counted: {total}) to {args.output}")


if __name__ == "__main__":
    main()
    # path = [
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-0SdAVK79lg.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-1LrH01Ei1w.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-4NLarMj4xU.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-5f6hjZf9Yw.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-5xOcMJpTUk.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-6QGvxvaTkI.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-6pcgdLfb_A.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-7wUQP6G5EQ.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-BIMKnb3tlo.wav",
    #     "/mnt/bn/audio-visual-llm-data/datasets/MusicCaps/Preprocess/-CUp_Tmg2Y0.wav",
    # ]
    
    # for p in path:
    #     new_p = change_musiccaps(p)
    #     if os.path.exists(new_p):
    #         print(f"Success: {new_p} exists.")
    #     else:
    #         print(f"Failure: {new_p} does not exist.")
