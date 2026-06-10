#!/usr/bin/env python3
"""Generate LibriSpeech volume-comparison MCQ samples in SALMONN format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf


TASK_NAME = "volume_comparison"
SOURCE_NAME = "volume_comparison_librispeech"
DEFAULT_LIBRISPEECH_DIR = Path("/data/milsrg1/corpora/librispeech/LibriSpeech/train-other-500")
DEFAULT_OUT_DIR = Path("/data/milsrg1/huggingface/cache/xy316/volume_comparison_librispeech")
DEFAULT_NUM_SAMPLES = 3000
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_MIN_SEGMENT_DURATION = 3.0
DEFAULT_MAX_SEGMENT_DURATION = 6.0
DEFAULT_PAUSE_DURATION = 0.1

PATTERNS = [
    "high-medium-low",
    "high-low-medium",
    "medium-high-low",
    "medium-low-high",
    "low-high-medium",
    "low-medium-high",
]


def discover_audio_files(librispeech_dir: Path) -> List[Path]:
    files = sorted(librispeech_dir.rglob("*.flac"))
    if not files:
        raise RuntimeError(f"No FLAC files found under {librispeech_dir}")
    return files


def resample_audio(audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
    if original_sr == target_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.zeros(0, dtype=np.float32)

    new_length = max(1, int(round(audio.size * target_sr / original_sr)))
    old_positions = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    new_positions = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(new_positions, old_positions, audio).astype(np.float32)


def trim_silence(audio: np.ndarray, sample_rate: int, top_db: float = 35.0) -> np.ndarray:
    if audio.size == 0:
        return audio

    frame_length = max(1, int(round(sample_rate * 0.02)))
    hop_length = max(1, int(round(sample_rate * 0.01)))
    if audio.size <= frame_length:
        return audio

    starts = list(range(0, audio.size - frame_length + 1, hop_length))
    if starts[-1] != audio.size - frame_length:
        starts.append(audio.size - frame_length)

    rms = np.asarray(
        [
            np.sqrt(np.mean(np.square(audio[start:start + frame_length]), dtype=np.float64))
            for start in starts
        ],
        dtype=np.float64,
    )
    peak_rms = float(np.max(rms)) if rms.size else 0.0
    if peak_rms <= 0.0:
        return audio

    threshold = peak_rms * (10.0 ** (-top_db / 20.0))
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return audio

    start = starts[int(active[0])]
    end = min(audio.size, starts[int(active[-1])] + frame_length)
    trimmed = audio[start:end]
    return trimmed if trimmed.size else audio


def read_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, always_2d=True)
    mono = audio.mean(axis=1).astype(np.float32)
    return resample_audio(mono, sample_rate, target_sr)


def choose_segment(
    audio: np.ndarray,
    sample_rate: int,
    min_duration: float,
    max_duration: float,
    rng: random.Random,
) -> Tuple[np.ndarray, float]:
    audio_duration = audio.size / sample_rate
    duration = rng.uniform(min_duration, min(max_duration, audio_duration))
    target_length = int(round(duration * sample_rate))
    if target_length <= 0:
        raise ValueError(f"Invalid segment duration: {duration}")
    if audio.size <= target_length:
        return audio, audio_duration

    start = rng.randint(0, audio.size - target_length)
    return audio[start:start + target_length], target_length / sample_rate


def build_level_gains(rng: random.Random, min_db: float, max_db: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    db_step = rng.uniform(min_db, max_db)
    gains = {
        "high": 10.0 ** (db_step / 20.0),
        "medium": 1.0,
        "low": 10.0 ** (-db_step / 20.0),
    }
    db_steps = {
        "low_to_medium_db": db_step,
        "medium_to_high_db": db_step,
        "low_to_high_db": db_step * 2.0,
    }
    return gains, db_steps


def make_volume_audio(
    segment: np.ndarray,
    pattern: str,
    gains: Dict[str, float],
    sample_rate: int,
    pause_duration: float,
) -> np.ndarray:
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    if peak <= 0.0:
        raise ValueError("Cannot build sample from silent audio")

    base = segment.astype(np.float32)
    pause_length = int(round(pause_duration * sample_rate))
    pause = np.zeros(pause_length, dtype=np.float32)

    pieces = []
    for index, level in enumerate(pattern.split("-")):
        if index > 0 and pause_length > 0:
            pieces.append(pause)
        pieces.append(base * gains[level])

    combined = np.concatenate(pieces).astype(np.float32)
    return np.clip(combined, -0.999, 0.999)


def build_options(pattern: str, rng: random.Random) -> Tuple[List[Dict[str, object]], str]:
    distractors = [candidate for candidate in PATTERNS if candidate != pattern]
    selected = rng.sample(distractors, 3) + [pattern]
    rng.shuffle(selected)

    options = []
    answer_label = ""
    for index, content in enumerate(selected):
        label = chr(ord("A") + index)
        is_correct = content == pattern
        if is_correct:
            answer_label = label
        options.append({"label": label, "content": content, "is_correct": is_correct})

    return options, answer_label


def format_prompt(question: str, options: Sequence[Dict[str, object]]) -> str:
    choices = "\n".join(
        f"Option {option['label']}: {option['content']}"
        for option in options
    )
    return (
        "<audio>Answer the following multiple-choice question using only the correct option.\n"
        f"Question: {question}\n"
        "Choices:\n"
        f"{choices}\n"
        "Please output your final answer with a single letter. For example, "
        "if you think the answer is Option A, please just output 'A'"
    )


def iter_source_files(files: Sequence[Path], rng: random.Random) -> Iterable[Path]:
    shuffled = list(files)
    while True:
        rng.shuffle(shuffled)
        yield from shuffled


def generate_dataset(args: argparse.Namespace) -> Dict[str, object]:
    rng = random.Random(args.seed)
    source_files = discover_audio_files(args.librispeech_dir)
    source_iter = iter_source_files(source_files, rng)

    audio_dir = args.out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    records = []
    skipped = 0
    attempts = 0
    max_attempts = args.num_samples * args.max_attempts_per_sample

    while len(records) < args.num_samples and attempts < max_attempts:
        attempts += 1
        source_path = next(source_iter)
        try:
            audio = trim_silence(read_audio(source_path, args.sample_rate), args.sample_rate)
            if audio.size < int(round(args.min_segment_duration * args.sample_rate)):
                skipped += 1
                continue

            segment, segment_duration = choose_segment(
                audio,
                args.sample_rate,
                args.min_segment_duration,
                args.max_segment_duration,
                rng,
            )
            gains, db_steps = build_level_gains(rng, args.min_db_step, args.max_db_step)
            pattern = rng.choice(PATTERNS)
            output_audio = make_volume_audio(
                segment,
                pattern,
                gains,
                args.sample_rate,
                args.pause_duration,
            )

            sample_id = f"{SOURCE_NAME}_{len(records):08d}"
            wav_path = audio_dir / f"{sample_id}.wav"
            sf.write(wav_path, output_audio, args.sample_rate)

            question = "Which volume pattern best matches the audio?"
            options, answer_label = build_options(pattern, rng)
            prompt = format_prompt(question, options)
            duration = round(float(output_audio.size) / args.sample_rate, 2)

            records.append(
                {
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer_label},
                    ],
                    "audios": [str(wav_path)],
                    "task_type": "qa_mc",
                    "task_name": TASK_NAME,
                    "source_task_name": SOURCE_NAME,
                    "mc_question": question,
                    "mc_options": options,
                    "mc_answer_label": answer_label,
                    "answer_gt": pattern,
                    "volume_pattern": pattern,
                    "volume_gains": gains,
                    "volume_db_steps": db_steps,
                    "pause_duration_sec": args.pause_duration,
                    "segment_duration_sec": round(segment_duration, 4),
                    "source_audio": str(source_path),
                    "durations": [duration],
                }
            )
        except Exception as exc:
            skipped += 1
            if args.verbose:
                print(f"Skipping {source_path}: {exc}")

    if len(records) < args.num_samples:
        raise RuntimeError(
            f"Only generated {len(records)} samples after {attempts} attempts; "
            f"skipped {skipped} source files."
        )

    return {
        "data": records,
        "metadata": {
            "task_name": TASK_NAME,
            "source_task_name": SOURCE_NAME,
            "source_split": "LibriSpeech/train-other-500",
            "librispeech_dir": str(args.librispeech_dir),
            "num_samples": len(records),
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "min_segment_duration": args.min_segment_duration,
            "max_segment_duration": args.max_segment_duration,
            "pause_duration": args.pause_duration,
            "min_db_step": args.min_db_step,
            "max_db_step": args.max_db_step,
            "attempts": attempts,
            "skipped": skipped,
            "patterns": PATTERNS,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-dir", type=Path, default=DEFAULT_LIBRISPEECH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=316)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-segment-duration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION)
    parser.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_DURATION)
    parser.add_argument("--pause-duration", type=float, default=DEFAULT_PAUSE_DURATION)
    parser.add_argument("--min-db-step", type=float, default=8.0)
    parser.add_argument("--max-db-step", type=float, default=12.0)
    parser.add_argument("--output-json-name", default="volume_comparison_librispeech_3k_salmonn_mcq.json")
    parser.add_argument("--max-attempts-per-sample", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_segment_duration <= 0.0:
        raise ValueError("--min-segment-duration must be positive")
    if args.max_segment_duration < args.min_segment_duration:
        raise ValueError("--max-segment-duration must be >= --min-segment-duration")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset = generate_dataset(args)

    output_json = args.out_dir / args.output_json_name
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(f"Wrote {len(dataset['data'])} samples to {output_json}")
    print(f"Wrote audio files to {args.out_dir / 'audio'}")


if __name__ == "__main__":
    main()
