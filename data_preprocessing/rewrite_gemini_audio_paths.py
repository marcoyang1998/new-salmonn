import argparse
import json
import os
from pathlib import Path


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


def change_path_librispeech(path: str) -> str:
    path = path.replace(
        "/mnt/bn/audio-visual-llm-data/yuwenyi/data/librispeech/LibriSpeech",
        "/mnt/shared-storage-user/brainllm-share/data/LibriSpeech",
    )
    path = path.replace(".wav", ".flac")
    return path


def change_path_gigaspeech(path: str) -> str:
    import pdb; pdb.set_trace()
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


def detect_dataset_name(audio_path: str) -> str | None:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rewrite audio paths in Gemini JSONL data.")
    parser.add_argument(
        "--input-jsonl",
        required=True,
        help="Path to the input JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional output JSONL path. Defaults to in-place rewrite.",
    )
    parser.add_argument(
        "--missing-log",
        default=None,
        help="Optional path to save rewritten records whose audio files do not exist.",
    )
    return parser.parse_args()


def rewrite_file(
    input_path: Path,
    output_path: Path,
    missing_log_path: Path | None,
) -> tuple[int, int, int, int, int]:
    total = 0
    rewritten = 0
    unchanged = 0
    existing = 0
    missing = 0

    missing_log = None
    if missing_log_path is not None:
        missing_log_path.parent.mkdir(parents=True, exist_ok=True)
        missing_log = missing_log_path.open("w", encoding="utf-8")

    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue

            total += 1
            record = json.loads(line)
            original_path = str(record.get("path", "")).strip()
            new_path = rewrite_audio_path(original_path)

            if new_path != original_path:
                rewritten += 1
                record["path"] = new_path
            else:
                unchanged += 1

            if new_path and os.path.exists(new_path):
                existing += 1
            else:
                missing += 1
                if missing_log is not None:
                    missing_log.write(json.dumps(record, ensure_ascii=False))
                    missing_log.write("\n")

            dst.write(json.dumps(record, ensure_ascii=False))
            dst.write("\n")

    if missing_log is not None:
        missing_log.close()

    return total, rewritten, unchanged, existing, missing


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl)
    missing_log_path = Path(args.missing_log) if args.missing_log is not None else None

    if args.output_jsonl is None:
        output_path = input_path.with_suffix(input_path.suffix + ".tmp")
        replace_in_place = True
    else:
        output_path = Path(args.output_jsonl)
        replace_in_place = False

    total, rewritten, unchanged, existing, missing = rewrite_file(
        input_path,
        output_path,
        missing_log_path,
    )

    if replace_in_place:
        output_path.replace(input_path)

    print(f"input_file={input_path}")
    print(f"output_file={input_path if replace_in_place else output_path}")
    print(f"total_records={total}")
    print(f"rewritten_paths={rewritten}")
    print(f"unchanged_paths={unchanged}")
    print(f"existing_audio_paths={existing}")
    print(f"missing_audio_paths={missing}")
    if missing_log_path is not None:
        print(f"missing_log={missing_log_path}")


if __name__ == "__main__":
    main()