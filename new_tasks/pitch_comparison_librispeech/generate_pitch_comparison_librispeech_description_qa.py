#!/usr/bin/env python3
"""Generate LibriSpeech pitch-comparison description QA samples in SALMONN format."""

from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf


TASK_NAME = "pitch_comparison"
SOURCE_NAME = "pitch_comparison_librispeech_description_qa"
DEFAULT_LIBRISPEECH_DIR = Path("/data/milsrg1/corpora/librispeech/LibriSpeech/test-other")
DEFAULT_AUDIO_PATH_PATTERN = "/mnt/shared-storage-user/brainllm-share/xiaoyu/gigaspeech/data/audio/l_files_additional/*/*.wav"
DEFAULT_OUT_DIR = Path("/data/milsrg1/huggingface/cache/xy316/pitch_comparison_librispeech_description_qa")
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

QUESTION_TEMPLATES = [
    {"id": "Q1", "text": "Describe how the pitch changes throughout the recording."},
    {"id": "Q2", "text": "How does the pitch vary across the three segments of the audio?"},
    {"id": "Q3", "text": "Summarize the pitch pattern in this recording."},
]

LEVEL_DESCRIPTIONS = {
    "low": "a low pitch",
    "medium": "a medium pitch",
    "high": "a high pitch",
}

LEVEL_RANKS = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def discover_audio_files(audio_path_pattern: str) -> List[Path]:
    if any(char in audio_path_pattern for char in "*?[]"):
        files = sorted(Path(path) for path in glob.glob(audio_path_pattern))
    else:
        audio_root = Path(audio_path_pattern)
        files = sorted(audio_root.rglob("*.flac")) + sorted(audio_root.rglob("*.wav"))
    if not files:
        raise RuntimeError(f"No audio files found for {audio_path_pattern}")
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


def sample_pitch_shifts(rng: random.Random) -> Dict[str, int]:
    return {
        "low": -rng.choice([2, 3]),
        "medium": 0,
        "high": rng.choice([2, 3]),
    }


def pitch_shift_parselmouth(audio: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
    if semitones == 0:
        return audio.astype(np.float32, copy=True)

    factor = 2.0 ** (semitones / 12.0)
    sound = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sample_rate)
    manipulation = call(sound, "To Manipulation", 0.01, 75, 600)
    pitch_tier = call(manipulation, "Extract pitch tier")
    call(pitch_tier, "Multiply frequencies", sound.xmin, sound.xmax, factor)
    call([pitch_tier, manipulation], "Replace pitch tier")
    shifted = call(manipulation, "Get resynthesis (overlap-add)")
    values = shifted.values[0].astype(np.float32)

    if values.size > audio.size:
        values = values[:audio.size]
    elif values.size < audio.size:
        values = np.pad(values, (0, audio.size - values.size))
    return values.astype(np.float32, copy=False)


def make_pitch_audio(
    segment: np.ndarray,
    pattern: str,
    pitch_shifts: Dict[str, int],
    sample_rate: int,
    pause_duration: float,
) -> np.ndarray:
    peak = float(np.max(np.abs(segment))) if segment.size else 0.0
    if peak <= 0.0:
        raise ValueError("Cannot build sample from silent audio")

    base = segment.astype(np.float32) / peak * 0.8
    pause_length = int(round(pause_duration * sample_rate))
    pause = np.zeros(pause_length, dtype=np.float32)

    shifted_by_level = {
        level: pitch_shift_parselmouth(base, sample_rate, semitones)
        for level, semitones in pitch_shifts.items()
    }

    pieces = []
    for index, level in enumerate(pattern.split("-")):
        if index > 0 and pause_length > 0:
            pieces.append(pause)
        pieces.append(shifted_by_level[level])

    combined = np.concatenate(pieces).astype(np.float32)
    peak = float(np.max(np.abs(combined))) if combined.size else 0.0
    if peak > 0.999:
        combined = combined / peak * 0.999
    return combined


def format_prompt(question: str) -> str:
    return f"<audio>{question}"


def describe_segment(index: int, level: str) -> str:
    if index == 0:
        return f"begins at {LEVEL_DESCRIPTIONS[level]}"
    if index == 1:
        return f"moves to {LEVEL_DESCRIPTIONS[level]} in the middle segment"
    return f"ends with {LEVEL_DESCRIPTIONS[level]} in the final segment"


def ordinal_segment_name(index: int) -> str:
    if index == 0:
        return "the first segment"
    if index == 1:
        return "the second segment"
    return "the third segment"


def build_segment_by_segment_answer(levels: Sequence[str], rng: random.Random) -> str:
    templates = [
        "The speech {first}, {second}, and {third}.",
        "It {first}, then {second}, before it {third}.",
        "The recording {first}, {second}, and {third}.",
    ]
    return rng.choice(templates).format(
        first=describe_segment(0, levels[0]),
        second=describe_segment(1, levels[1]),
        third=describe_segment(2, levels[2]),
    )


def build_relative_comparison_answer(levels: Sequence[str], rng: random.Random) -> str:
    highest_index = max(range(3), key=lambda idx: LEVEL_RANKS[levels[idx]])
    lowest_index = min(range(3), key=lambda idx: LEVEL_RANKS[levels[idx]])
    intermediate_index = ({0, 1, 2} - {highest_index, lowest_index}).pop()

    highest = ordinal_segment_name(highest_index)
    lowest = ordinal_segment_name(lowest_index)
    intermediate = ordinal_segment_name(intermediate_index)
    templates = [
        (
            "{highest_cap} has the highest pitch, {intermediate} is intermediate, "
            "and {lowest} has the lowest pitch."
        ),
        (
            "The highest-pitched part is {highest}, while {lowest} has the lowest pitch; "
            "{intermediate} falls between them."
        ),
        (
            "{lowest_cap} has the lowest pitch, {intermediate} is in the middle, "
            "and {highest} has the highest pitch."
        ),
    ]
    return rng.choice(templates).format(
        highest=highest,
        highest_cap=highest.capitalize(),
        intermediate=intermediate,
        lowest=lowest,
        lowest_cap=lowest.capitalize(),
    )


def build_trend_answer(levels: Sequence[str], rng: random.Random) -> str:
    templates = [
        "The pitch changes from {first} to {second}, then to {third}.",
        "The pitch moves from {first} into {second} and finishes at {third}.",
        "Across the recording, the pitch goes {first}, then {second}, then {third}.",
    ]
    return rng.choice(templates).format(first=levels[0], second=levels[1], third=levels[2])


def build_answer(pattern: str, rng: random.Random) -> Tuple[str, str]:
    levels = pattern.split("-")
    style = rng.choice(["A", "B", "C"])
    if style == "A":
        return build_segment_by_segment_answer(levels, rng), style
    if style == "B":
        return build_relative_comparison_answer(levels, rng), style
    return build_trend_answer(levels, rng), style


def iter_source_files(files: Sequence[Path], rng: random.Random) -> Iterable[Path]:
    shuffled = list(files)
    while True:
        rng.shuffle(shuffled)
        yield from shuffled


def generate_dataset(args: argparse.Namespace) -> Dict[str, object]:
    rng = random.Random(args.seed)
    source_files = discover_audio_files(args.audio_path_pattern)
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
            pitch_shifts = sample_pitch_shifts(rng)
            pattern = rng.choice(PATTERNS)
            output_audio = make_pitch_audio(
                segment,
                pattern,
                pitch_shifts,
                args.sample_rate,
                args.pause_duration,
            )

            sample_id = f"{SOURCE_NAME}_{len(records):08d}"
            wav_path = audio_dir / f"{sample_id}.wav"
            sf.write(wav_path, output_audio, args.sample_rate)

            question_template = rng.choice(QUESTION_TEMPLATES)
            question = question_template["text"]
            answer, answer_style = build_answer(pattern, rng)
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
                    "question_template_id": question_template["id"],
                    "answer_style": answer_style,
                    "answer_gt": pattern,
                    "pitch_pattern": pattern,
                    "pitch_shifts_semitones": pitch_shifts,
                    "pitch_tool": "parselmouth_praat_overlap_add",
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
            "source_split": "custom_audio_pattern",
            "audio_path_pattern": args.audio_path_pattern,
            "num_samples": len(records),
            "seed": args.seed,
            "sample_rate": args.sample_rate,
            "min_segment_duration": args.min_segment_duration,
            "max_segment_duration": args.max_segment_duration,
            "pause_duration": args.pause_duration,
            "pitch_shift_choices": [-3, -2, 0, 2, 3],
            "pitch_tool": "parselmouth_praat_overlap_add",
            "attempts": attempts,
            "skipped": skipped,
            "patterns": PATTERNS,
            "question_templates": QUESTION_TEMPLATES,
            "answer_styles": {
                "A": "Segment-by-segment description",
                "B": "Relative comparison",
                "C": "Trend description",
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-path-pattern", default=DEFAULT_AUDIO_PATH_PATTERN)
    parser.add_argument(
        "--librispeech-dir",
        dest="audio_path_pattern",
        default=argparse.SUPPRESS,
        help="Backward-compatible alias for --audio-path-pattern.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=616)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-segment-duration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION)
    parser.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_DURATION)
    parser.add_argument("--pause-duration", type=float, default=DEFAULT_PAUSE_DURATION)
    parser.add_argument("--output-json-name", default="pitch_comparison_librispeech_description_qa_10k.json")
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
