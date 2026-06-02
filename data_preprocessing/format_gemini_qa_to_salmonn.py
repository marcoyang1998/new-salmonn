import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path
from typing import Optional

import torchaudio


YOUTUBE_IDS_PATH = (
    "/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/"
    "general_audio_encoder/mtl/data/audioset_manifest/youtube_ids_all.txt"
)


def load_audioset_youtube_ids() -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        with open(YOUTUBE_IDS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ytid, audio = line.split(maxsplit=1)
                mapping[ytid] = audio
    except Exception:
        pass
    return mapping


AUDIOSSET_YOUTUBE_IDS = load_audioset_youtube_ids()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format Gemini QA data into SALMONN format.")
    parser.add_argument(
        "--input-json",
        default="salmonn_data_v1.1/gemini_qa.json",
        help="Input Gemini QA JSON path.",
    )
    parser.add_argument(
        "--output-json",
        default="salmonn_data_v1.1/stage2/gemini_qa_formatted.json",
        help="Output formatted JSON path.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) // 2),
        help="Number of worker processes for audio validation and conversion.",
    )
    return parser.parse_args()


def convert_entry(entry: dict) -> dict:
    question = str(entry.get("Q", "")).strip()
    answer = str(entry.get("text", "")).strip()
    audio_path = str(entry.get("path", "")).strip()
    task_type = str(entry.get("task", "QA")).strip() or "QA"

    return {
        "messages": [
            {
                "role": "user",
                "content": f"<audio>{question}",
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ],
        "audios": [audio_path],
        "task_type": task_type,
    }


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
            return "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/xs_files/" + path
        if subset == "s":
            return "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/s_files_additional/" + path
        if subset == "m":
            return "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/m_files_additional/" + path
        if subset == "subset_test":
            return "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/test_files/" + path
        raise ValueError(f"Unexpected GigaSpeech subset in path: {path}")

    wav_file = path.split("/")[-1]
    for chunk in [0, 1, 2]:
        potential_path = (
            f"/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/"
            f"test_files/test_chunks_000{chunk}/{wav_file}"
        )
        if os.path.exists(potential_path):
            return potential_path
    return path


def change_path_audiocaps(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/yuwenyi/audiocaps/audiocaps/",
        "/mnt/shared-storage-user/brainllm-share/data/AudioCaps/audio/",
    )
    dir_name = os.path.dirname(path)
    filename = os.path.basename(path)
    return os.path.join(dir_name, "Y" + filename)


def change_path_iemocap(path: str) -> str:
    return path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/iemocap/IEMOCAP_full_release/",
        "/mnt/shared-storage-user/brainllm-share/data/IEMOCAP/",
    )


def change_path_voxceleb(path: str) -> str:
    return path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/Voxceleb1/",
        "/mnt/shared-storage-user/brainllm-share/data/Voxceleb1/",
    )


def change_path_librimix(path: str) -> str:
    return path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/LibriMix/mixdata/Libri2Mix/",
        "/mnt/shared-storage-user/brainllm-share/xiaoyu/Libri2Mix/",
    )


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
    return new_path.replace(".wav", ".flac")


def change_musiccaps(path: str) -> str:
    filename = os.path.basename(path)
    ytid = filename.replace(".wav", "")
    return AUDIOSSET_YOUTUBE_IDS.get(ytid, "")


def change_millionsong(path: str) -> str:
    return path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/MillionSongDatasetSpotify/",
        "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/MillionSongDatasetSpotify/",
    )


def change_musicnet(path: str) -> str:
    return path.replace(
        "/mnt/bn/audio-visual-llm-data/datasets/MusicNet/Preprocess/train/",
        "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/musicnet/train_data/",
    )


def detect_dataset_name(audio_path: str) -> Optional[str]:
    audio_lower = audio_path.lower()
    for name in [
        "librispeech",
        "gigaspeech",
        "audiocaps",
        "iemocap",
        "voxceleb1",
        "librimix",
        "clotho",
        "commonvoice",
        "wavcaps",
        "musiccaps",
        "millionsong",
        "musicnet",
    ]:
        if name in audio_lower:
            return name
    return None


def rewrite_audio_path(audio_path: str) -> str:
    name = detect_dataset_name(audio_path)
    if name == "librispeech":
        return change_path_librispeech(audio_path)
    if name == "gigaspeech":
        return change_path_gigaspeech(audio_path)
    if name == "audiocaps":
        return change_path_audiocaps(audio_path)
    if name == "iemocap":
        return change_path_iemocap(audio_path)
    if name == "voxceleb1":
        return change_path_voxceleb(audio_path)
    if name == "librimix":
        return change_path_librimix(audio_path)
    if name == "clotho":
        return change_path_clotho(audio_path)
    if name == "commonvoice":
        return change_commonvoice(audio_path)
    if name == "wavcaps":
        return change_wavcaps(audio_path)
    if name == "musiccaps":
        return change_musiccaps(audio_path)
    if name == "millionsong":
        return change_millionsong(audio_path)
    if name == "musicnet":
        return change_musicnet(audio_path)
    return audio_path


def is_audio_valid(audio_path: str) -> bool:
    if not audio_path:
        return False
    if not os.path.exists(audio_path):
        return False
    try:
        torchaudio.info(audio_path)
        return True
    except Exception:
        return False


def convert_entry_if_valid(entry: dict) -> tuple[dict | None, str | None]:
    original_audio_path = str(entry.get("path", "")).strip()
    rewritten_audio_path = rewrite_audio_path(original_audio_path)

    if not rewritten_audio_path:
        return None, f"WARNING: empty rewritten path from original path: {original_audio_path}"

    if not is_audio_valid(rewritten_audio_path):
        return None, (
            f"WARNING: invalid audio after rewrite. "
            f"original={original_audio_path} rewritten={rewritten_audio_path}"
        )

    new_entry = dict(entry)
    new_entry["path"] = rewritten_audio_path
    return convert_entry(new_entry), None


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected top-level JSON array in gemini_qa.json")

    num_workers = max(1, args.num_workers)
    with mp.Pool(processes=num_workers) as pool:
        converted = pool.imap(convert_entry_if_valid, data, chunksize=256)
        formatted = []
        warnings = []
        warning_reason_counts: Counter = Counter()
        for item, warning_msg in converted:
            if item is not None:
                formatted.append(item)
            else:
                if warning_msg is not None:
                    warnings.append(warning_msg)
                    if "empty rewritten path" in warning_msg:
                        warning_reason_counts["empty_rewritten_path"] += 1
                    elif "invalid audio after rewrite" in warning_msg:
                        warning_reason_counts["invalid_or_missing_audio_after_rewrite"] += 1
                    else:
                        warning_reason_counts["other"] += 1

    skipped = len(data) - len(formatted)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"data": formatted}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"input_count={len(data)}")
    print(f"output_count={len(formatted)}")
    print(f"skipped_invalid_audio={skipped}")
    print(f"num_workers={num_workers}")
    print(f"saved_to={output_path}")

    if warning_reason_counts:
        print("warning_summary:")
        for reason, count in sorted(warning_reason_counts.items()):
            print(f"  {reason}={count}")

    if warnings:
        print("sample_warnings:")
        for line in warnings[:20]:
            print(line)


if __name__ == "__main__":
    main()