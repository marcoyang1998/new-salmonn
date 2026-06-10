#!/usr/bin/env python3
"""Generate targeted SPGISpeech speed-comparison QA samples in SALMONN format."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import soundfile as sf


TASK_NAME = "speed_comparison"
SOURCE_NAME = "speed_comparison_spgispeech_targeted_qa"
DEFAULT_SPGISPEECH_DIR = Path("/scratches/million/xy316/spgispeech/S")
DEFAULT_OUT_DIR = Path("/data/milsrg1/huggingface/cache/xy316/speed_comparison_spgispeech_targeted_qa")
DEFAULT_NUM_SAMPLES = 10000
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

QUESTION_TEMPLATES = {
    "fastest": [
        "Which segment has the highest speaking rate?",
        "Which part of the recording is spoken the fastest?",
    ],
    "slowest": [
        "Which segment has the lowest speaking rate?",
        "Which part of the recording is spoken the slowest?",
    ],
    "relative": [
        "Which segment is spoken faster, the {x} or the {y}?",
        "Is the {x} segment faster than the {y} segment?",
    ],
}

LEVEL_RANKS = {
    "low": 0,
    "medium": 1,
    "high": 2,
}

ORDINALS = {
    0: "first",
    1: "second",
    2: "third",
}

POSITION_NAMES = {
    0: "beginning",
    1: "middle",
    2: "final",
}


def discover_audio_files(spgispeech_dir: Path) -> List[Path]:
    files = sorted((spgispeech_dir / "wav").rglob("*.wav"))
    if not files:
        files = sorted(spgispeech_dir.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No WAV files found under {spgispeech_dir}")
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


def sample_speed_rates(
    rng: random.Random,
    min_low_speed: float,
    max_low_speed: float,
    min_high_speed: float,
    max_high_speed: float,
) -> Dict[str, float]:
    return {
        "low": rng.uniform(min_low_speed, max_low_speed),
        "medium": 1.0,
        "high": rng.uniform(min_high_speed, max_high_speed),
    }


def choose_medium_segment(
    audio: np.ndarray,
    sample_rate: int,
    speed_rates: Dict[str, float],
    min_duration: float,
    max_duration: float,
    rng: random.Random,
) -> Tuple[np.ndarray, float]:
    audio_duration = audio.size / sample_rate
    min_medium_duration = max(min_duration, min_duration * speed_rates["high"])
    max_medium_duration = min(max_duration, max_duration * speed_rates["low"], audio_duration)

    if max_medium_duration < min_medium_duration:
        raise ValueError(
            "Source audio is too short for the sampled speed rates while keeping every segment "
            f"within {min_duration}-{max_duration}s."
        )

    duration = rng.uniform(min_medium_duration, max_medium_duration)
    target_length = int(round(duration * sample_rate))
    if target_length <= 0:
        raise ValueError(f"Invalid segment duration: {duration}")
    if audio.size <= target_length:
        return audio, audio_duration

    start = rng.randint(0, audio.size - target_length)
    return audio[start:start + target_length], target_length / sample_rate


def change_speed_resample(audio: np.ndarray, speed_rate: float) -> np.ndarray:
    if speed_rate <= 0.0:
        raise ValueError(f"Speed rate must be positive: {speed_rate}")
    if audio.size == 0:
        return audio.astype(np.float32)

    output_length = max(1, int(round(audio.size / speed_rate)))
    source_positions = np.arange(output_length, dtype=np.float64) * speed_rate
    source_positions = np.minimum(source_positions, audio.size - 1)
    old_positions = np.arange(audio.size, dtype=np.float64)
    return np.interp(source_positions, old_positions, audio).astype(np.float32)


def change_speed_pitch_preserving(audio: np.ndarray, speed_rate: float) -> np.ndarray:
    if speed_rate <= 0.0:
        raise ValueError(f"Speed rate must be positive: {speed_rate}")
    if audio.size == 0:
        return audio.astype(np.float32)
    if abs(speed_rate - 1.0) < 1e-6:
        return audio.astype(np.float32, copy=True)

    numba_cache_dir = Path(tempfile.gettempdir()) / "numba_cache"
    numba_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(numba_cache_dir))
    import librosa

    stretched = librosa.effects.time_stretch(
        audio.astype(np.float32, copy=False),
        rate=speed_rate,
    )

    peak = float(np.max(np.abs(stretched))) if stretched.size else 0.0
    if peak <= 0.0:
        raise ValueError("Pitch-preserving speed perturbation produced empty audio")
    if peak > 1.0:
        stretched = stretched / peak
    return stretched.astype(np.float32, copy=False)


def change_speed(audio: np.ndarray, speed_rate: float, method: str) -> np.ndarray:
    if method == "resample":
        return change_speed_resample(audio, speed_rate)
    if method == "pitch_preserving":
        return change_speed_pitch_preserving(audio, speed_rate)
    raise ValueError(f"Unsupported speed perturbation method: {method}")


def make_speed_audio(
    segment: np.ndarray,
    pattern: str,
    speed_rates: Dict[str, float],
    sample_rate: int,
    pause_duration: float,
    speed_perturbation_method: str,
) -> Tuple[np.ndarray, Dict[str, float]]:
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    if peak <= 0.0:
        raise ValueError("Cannot build sample from silent audio")

    pause_length = int(round(pause_duration * sample_rate))
    pause = np.zeros(pause_length, dtype=np.float32)
    pieces = []
    level_durations = {}

    for index, level in enumerate(pattern.split("-")):
        if index > 0 and pause_length > 0:
            pieces.append(pause)
        changed = change_speed(segment, speed_rates[level], speed_perturbation_method)
        pieces.append(changed)
        level_durations[level] = round(changed.size / sample_rate, 4)

    combined = np.concatenate(pieces).astype(np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.999:
        combined = combined / peak * 0.999
    return combined, level_durations


def format_prompt(question: str) -> str:
    return f"<audio>{question}"


def segment_phrase(index: int) -> str:
    return f"the {ORDINALS[index]} segment"


def part_phrase(index: int) -> str:
    return f"the {POSITION_NAMES[index]} part"


def build_fastest_qa(levels: Sequence[str], rng: random.Random) -> Tuple[str, str, Dict[str, object]]:
    target_index = levels.index("high")
    question = rng.choice(QUESTION_TEMPLATES["fastest"])
    answer = rng.choice(
        [
            f"{segment_phrase(target_index).capitalize()} has the highest speaking rate.",
            f"{part_phrase(target_index).capitalize()} is spoken the fastest.",
            f"The fastest portion is {segment_phrase(target_index)}.",
        ]
    )
    return question, answer, {"target_segment_index": target_index + 1}


def build_slowest_qa(levels: Sequence[str], rng: random.Random) -> Tuple[str, str, Dict[str, object]]:
    target_index = levels.index("low")
    question = rng.choice(QUESTION_TEMPLATES["slowest"])
    answer = rng.choice(
        [
            f"{segment_phrase(target_index).capitalize()} has the lowest speaking rate.",
            f"{part_phrase(target_index).capitalize()} is spoken the slowest.",
            f"The slowest portion is {segment_phrase(target_index)}.",
        ]
    )
    return question, answer, {"target_segment_index": target_index + 1}


def build_relative_qa(levels: Sequence[str], rng: random.Random) -> Tuple[str, str, Dict[str, object]]:
    x_index, y_index = rng.sample(range(3), 2)
    x_level = levels[x_index]
    y_level = levels[y_index]
    x_is_faster = LEVEL_RANKS[x_level] > LEVEL_RANKS[y_level]
    faster_index = x_index if x_is_faster else y_index

    template = rng.choice(QUESTION_TEMPLATES["relative"])
    question = template.format(x=ORDINALS[x_index], y=ORDINALS[y_index])
    if template.startswith("Is"):
        if x_is_faster:
            answer = f"Yes. {segment_phrase(x_index).capitalize()} is spoken faster than {segment_phrase(y_index)}."
        else:
            answer = f"No. {segment_phrase(y_index).capitalize()} is spoken faster than {segment_phrase(x_index)}."
    else:
        answer = rng.choice(
            [
                f"{segment_phrase(faster_index).capitalize()} is spoken faster.",
                f"The faster one is {segment_phrase(faster_index)}.",
                f"{segment_phrase(faster_index).capitalize()} has the higher speaking rate.",
            ]
        )

    return question, answer, {
        "x_segment_index": x_index + 1,
        "y_segment_index": y_index + 1,
        "x_is_faster": x_is_faster,
        "target_segment_index": faster_index + 1,
    }


def build_question_answer(pattern: str, rng: random.Random) -> Tuple[str, str, str, Dict[str, object]]:
    levels = pattern.split("-")
    question_type = rng.choice(["fastest", "slowest", "relative"])
    if question_type == "fastest":
        question, answer, metadata = build_fastest_qa(levels, rng)
    elif question_type == "slowest":
        question, answer, metadata = build_slowest_qa(levels, rng)
    else:
        question, answer, metadata = build_relative_qa(levels, rng)
    return question, answer, question_type, metadata


def iter_source_files(files: Sequence[Path], rng: random.Random) -> Iterable[Path]:
    shuffled = list(files)
    while True:
        rng.shuffle(shuffled)
        yield from shuffled


def generate_dataset(args: argparse.Namespace) -> Dict[str, object]:
    rng = random.Random(args.seed)
    source_files = discover_audio_files(args.spgispeech_dir)
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
            speed_rates = sample_speed_rates(
                rng,
                args.min_low_speed,
                args.max_low_speed,
                args.min_high_speed,
                args.max_high_speed,
            )
            audio = trim_silence(read_audio(source_path, args.sample_rate), args.sample_rate)
            segment, medium_segment_duration = choose_medium_segment(
                audio,
                args.sample_rate,
                speed_rates,
                args.min_segment_duration,
                args.max_segment_duration,
                rng,
            )
            pattern = rng.choice(PATTERNS)
            output_audio, speed_segment_durations = make_speed_audio(
                segment,
                pattern,
                speed_rates,
                args.sample_rate,
                args.pause_duration,
                args.speed_perturbation_method,
            )

            sample_id = f"{SOURCE_NAME}_{len(records):08d}"
            wav_path = audio_dir / f"{sample_id}.wav"
            sf.write(wav_path, output_audio, args.sample_rate)

            question, answer, question_type, question_metadata = build_question_answer(pattern, rng)
            duration = round(float(output_audio.size) / args.sample_rate, 2)

            records.append(
                {
                    "messages": [
                        {"role": "user", "content": format_prompt(question)},
                        {"role": "assistant", "content": answer},
                    ],
                    "audios": [str(wav_path)],
                    "task_type": "qa",
                    "task_name": TASK_NAME,
                    "source_task_name": SOURCE_NAME,
                    "question": question,
                    "answer": answer,
                    "question_type": question_type,
                    "question_metadata": question_metadata,
                    "answer_gt": pattern,
                    "speed_pattern": pattern,
                    "speed_rates": speed_rates,
                    "speed_perturbation_method": args.speed_perturbation_method,
                    "pause_duration_sec": args.pause_duration,
                    "medium_source_segment_duration_sec": round(medium_segment_duration, 4),
                    "speed_segment_durations_sec": speed_segment_durations,
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
            "source_split": "SPGISpeech/S",
            "spgispeech_dir": str(args.spgispeech_dir),
            "num_samples": len(records),
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "min_segment_duration": args.min_segment_duration,
            "max_segment_duration": args.max_segment_duration,
            "pause_duration": args.pause_duration,
            "min_low_speed": args.min_low_speed,
            "max_low_speed": args.max_low_speed,
            "min_high_speed": args.min_high_speed,
            "max_high_speed": args.max_high_speed,
            "speed_perturbation_method": args.speed_perturbation_method,
            "attempts": attempts,
            "skipped": skipped,
            "patterns": PATTERNS,
            "question_templates": QUESTION_TEMPLATES,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spgispeech-dir", type=Path, default=DEFAULT_SPGISPEECH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=316)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-segment-duration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION)
    parser.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_DURATION)
    parser.add_argument("--pause-duration", type=float, default=DEFAULT_PAUSE_DURATION)
    parser.add_argument("--min-low-speed", type=float, default=0.85)
    parser.add_argument("--max-low-speed", type=float, default=0.90)
    parser.add_argument("--min-high-speed", type=float, default=1.15)
    parser.add_argument("--max-high-speed", type=float, default=1.25)
    parser.add_argument(
        "--speed-perturbation-method",
        choices=("pitch_preserving", "resample"),
        default="pitch_preserving",
        help=(
            "pitch_preserving changes speaking rate with librosa time-stretching; "
            "resample changes pitch together with speed."
        ),
    )
    parser.add_argument("--output-json-name", default="speed_comparison_spgispeech_targeted_qa_10k.json")
    parser.add_argument("--max-attempts-per-sample", type=int, default=5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_segment_duration <= 0.0:
        raise ValueError("--min-segment-duration must be positive")
    if args.max_segment_duration < args.min_segment_duration:
        raise ValueError("--max-segment-duration must be >= --min-segment-duration")
    if not (0.0 < args.min_low_speed <= args.max_low_speed < 1.0):
        raise ValueError("Expected 0 < min-low-speed <= max-low-speed < 1")
    if not (1.0 < args.min_high_speed <= args.max_high_speed):
        raise ValueError("Expected 1 < min-high-speed <= max-high-speed")

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
