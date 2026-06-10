#!/usr/bin/env python3
"""Generate targeted LibriSpeech volume-comparison QA samples in SALMONN format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Sequence, Tuple

import soundfile as sf

from generate_volume_comparison_librispeech_description_qa import (
    DEFAULT_LIBRISPEECH_DIR,
    DEFAULT_MAX_SEGMENT_DURATION,
    DEFAULT_MIN_SEGMENT_DURATION,
    DEFAULT_PAUSE_DURATION,
    DEFAULT_SAMPLE_RATE,
    LEVEL_RANKS,
    PATTERNS,
    TASK_NAME,
    build_level_gains,
    choose_segment,
    discover_audio_files,
    format_prompt,
    iter_source_files,
    make_volume_audio,
    read_audio,
    trim_silence,
)


SOURCE_NAME = "volume_comparison_librispeech_targeted_qa"
DEFAULT_OUT_DIR = Path("/data/milsrg1/huggingface/cache/xy316/volume_comparison_librispeech_targeted_qa")
DEFAULT_NUM_SAMPLES = 10000

QUESTION_TEMPLATES = {
    "loudest": [
        "Which segment has the highest volume?",
        "Which part of the recording is the loudest?",
    ],
    "quietest": [
        "Which segment has the lowest volume?",
        "Which part of the recording is the quietest?",
    ],
    "relative": [
        "Which segment is louder, the {x} or the {y}?",
        "Is the {x} segment louder than the {y} segment?",
    ],
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


def segment_phrase(index: int) -> str:
    return f"the {ORDINALS[index]} segment"


def part_phrase(index: int) -> str:
    return f"the {POSITION_NAMES[index]} part"


def build_loudest_qa(levels: Sequence[str], rng: random.Random) -> Tuple[str, str, Dict[str, object]]:
    target_index = levels.index("high")
    question = rng.choice(QUESTION_TEMPLATES["loudest"])
    answer = rng.choice(
        [
            f"{segment_phrase(target_index).capitalize()} has the highest volume.",
            f"{part_phrase(target_index).capitalize()} is the loudest.",
            f"The loudest portion is {segment_phrase(target_index)}.",
        ]
    )
    return question, answer, {"target_segment_index": target_index + 1}


def build_quietest_qa(levels: Sequence[str], rng: random.Random) -> Tuple[str, str, Dict[str, object]]:
    target_index = levels.index("low")
    question = rng.choice(QUESTION_TEMPLATES["quietest"])
    answer = rng.choice(
        [
            f"{segment_phrase(target_index).capitalize()} has the lowest volume.",
            f"{part_phrase(target_index).capitalize()} is the quietest.",
            f"The quietest portion is {segment_phrase(target_index)}.",
        ]
    )
    return question, answer, {"target_segment_index": target_index + 1}


def build_relative_qa(levels: Sequence[str], rng: random.Random) -> Tuple[str, str, Dict[str, object]]:
    x_index, y_index = rng.sample(range(3), 2)
    x_level = levels[x_index]
    y_level = levels[y_index]
    x_is_louder = LEVEL_RANKS[x_level] > LEVEL_RANKS[y_level]
    louder_index = x_index if x_is_louder else y_index

    template = rng.choice(QUESTION_TEMPLATES["relative"])
    question = template.format(x=ORDINALS[x_index], y=ORDINALS[y_index])
    if template.startswith("Is"):
        if x_is_louder:
            answer = f"Yes. {segment_phrase(x_index).capitalize()} is louder than {segment_phrase(y_index)}."
        else:
            answer = f"No. {segment_phrase(y_index).capitalize()} is louder than {segment_phrase(x_index)}."
    else:
        answer = rng.choice(
            [
                f"{segment_phrase(louder_index).capitalize()} is louder.",
                f"The louder one is {segment_phrase(louder_index)}.",
                f"{segment_phrase(louder_index).capitalize()} has the higher volume.",
            ]
        )

    return question, answer, {
        "x_segment_index": x_index + 1,
        "y_segment_index": y_index + 1,
        "x_is_louder": x_is_louder,
        "target_segment_index": louder_index + 1,
    }


def build_question_answer(pattern: str, rng: random.Random) -> Tuple[str, str, str, Dict[str, object]]:
    levels = pattern.split("-")
    question_type = rng.choice(["loudest", "quietest", "relative"])
    if question_type == "loudest":
        question, answer, metadata = build_loudest_qa(levels, rng)
    elif question_type == "quietest":
        question, answer, metadata = build_quietest_qa(levels, rng)
    else:
        question, answer, metadata = build_relative_qa(levels, rng)
    return question, answer, question_type, metadata


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
            "question_templates": QUESTION_TEMPLATES,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--librispeech-dir", type=Path, default=DEFAULT_LIBRISPEECH_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--seed", type=int, default=417)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-segment-duration", type=float, default=DEFAULT_MIN_SEGMENT_DURATION)
    parser.add_argument("--max-segment-duration", type=float, default=DEFAULT_MAX_SEGMENT_DURATION)
    parser.add_argument("--pause-duration", type=float, default=DEFAULT_PAUSE_DURATION)
    parser.add_argument("--min-db-step", type=float, default=8.0)
    parser.add_argument("--max-db-step", type=float, default=12.0)
    parser.add_argument("--output-json-name", default="volume_comparison_librispeech_targeted_qa_10k.json")
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
