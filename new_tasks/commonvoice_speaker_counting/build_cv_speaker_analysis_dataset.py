#!/usr/bin/env python3
"""
Build Common Voice speaker-analysis QA and MCQ datasets.

The script expects Common Voice speaker metadata with self-reported speaker
attributes. It can use either a rich metadata JSON that already contains audio
paths/transcripts, or a metadata JSON plus an existing SALMONN ASR JSON to fill
missing audio/transcript/duration fields.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm


DEFAULT_OUT_DIR = Path("salmonn_data_v1.1/speaker_analysis")
DEFAULT_QA_OUT = DEFAULT_OUT_DIR / "cv17_speaker_analysis_qa_25k.json"
DEFAULT_MCQ_OUT = DEFAULT_OUT_DIR / "cv17_speaker_analysis_mcq_25k.json"
DEFAULT_METADATA_OUT = DEFAULT_OUT_DIR / "speaker_analysis_metadata.json"
DEFAULT_REPORT_OUT = DEFAULT_OUT_DIR / "speaker_analysis_report.json"

EPS = 1e-10
MC_FINAL_INSTRUCTION = (
    "Please output your final answer with a single letter. "
    "For example, if you think the answer is Option A, please just output 'A'"
)

FIELD_CANDIDATES = {
    "speaker_id": ("speaker_id", "client_id", "clientId", "speaker", "user_id", "user"),
    "audio_path": ("audio_path", "audio", "path", "wav", "file", "filename", "location", "mp3"),
    "transcript": ("transcript", "sentence", "text", "normalized_text", "utterance"),
    "duration": ("duration", "dur", "audio_duration", "length"),
    "gender": ("gender", "sex"),
    "age": ("age", "age_group", "agegroup"),
    "accent": ("accent", "accents", "accent_group", "accent_group_id"),
}

GENDER_DISPLAY = {
    "m": "male",
    "man": "male",
    "male": "male",
    "male_masculine": "male",
    "masculine": "male",
    "f": "female",
    "woman": "female",
    "female": "female",
    "female_feminine": "female",
    "feminine": "female",
}

MISSING_VALUES = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "unknown",
    "unspecified",
    "not specified",
    "other",
}

QUESTION_TYPE_WEIGHTS = (
    ("single_attribute", 0.20),
    ("gender_count", 0.30),
    ("ordinal_attribute", 0.30),
    ("keyword_attribute", 0.20),
)

ATTRIBUTES = ("gender", "age", "accent")
ORDINALS = ("first", "second", "last")
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class SourceUtterance:
    audio_path: str
    speaker_id: str
    transcript: str
    duration: Optional[float]
    gender: Optional[str]
    age: Optional[str]
    accent: Optional[str]

    def label(self, attribute: str) -> Optional[str]:
        return getattr(self, attribute)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def unwrap_records(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in ("data", "records", "items", "metadata", "clips"):
            value = payload.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            records = []
            for key, value in payload.items():
                record = dict(value)
                record.setdefault("speaker_id", key)
                records.append(record)
            return records
    raise ValueError("Expected JSON list, {'data': [...]}, or dict of records.")


def expand_metadata_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    expanded = []
    child_keys = ("clips", "utterances", "samples", "segments")
    for record in records:
        nested_key = next((key for key in child_keys if isinstance(record.get(key), list)), None)
        if nested_key is None:
            expanded.append(record)
            continue

        base = {key: value for key, value in record.items() if key != nested_key}
        for child in record[nested_key]:
            merged = dict(base)
            if isinstance(child, dict):
                merged.update(child)
            else:
                merged["audio_path"] = child
            expanded.append(merged)
    return expanded


def first_present(record: Dict[str, Any], names: Sequence[str]) -> Optional[Any]:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    lowered = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is not None:
            return value
    return None


def normalize_text_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        value = ", ".join(str(v).strip() for v in value if str(v).strip())
    text = str(value).strip()
    if text.lower() in MISSING_VALUES:
        return None
    return text


def normalize_gender(value: Any) -> Optional[str]:
    text = normalize_text_value(value)
    if text is None:
        return None
    key = text.lower().replace(" ", "_").replace("-", "_")
    return GENDER_DISPLAY.get(key, text.replace("_", " "))


def normalize_label(attribute: str, value: Any) -> Optional[str]:
    if attribute == "gender":
        return normalize_gender(value)
    text = normalize_text_value(value)
    if text is None:
        return None
    return text.replace("_", " ") if attribute == "accent" else text


def as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def extract_assistant_text(messages: Any) -> Optional[str]:
    if not isinstance(messages, list):
        return None
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "assistant":
            return normalize_text_value(message.get("content"))
    return None


def build_salmonn_index(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None:
        return {}
    payload = read_json(path)
    records = unwrap_records(payload)
    index: Dict[str, Dict[str, Any]] = {}
    for item in records:
        audios = item.get("audios")
        if not isinstance(audios, list) or not audios:
            continue
        audio_path = str(audios[0])
        transcript = extract_assistant_text(item.get("messages"))
        duration = None
        durations = item.get("durations")
        if isinstance(durations, list) and durations:
            duration = as_float(durations[0])
        payload_item = {
            "audio_path": audio_path,
            "transcript": transcript,
            "duration": duration,
        }
        for key in audio_keys(audio_path):
            index.setdefault(key, payload_item)
    return index


def load_salmonn_records(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None:
        return []
    return unwrap_records(read_json(path))


def build_attr_by_speaker(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[str]]]:
    attrs_by_speaker: Dict[str, Dict[str, Optional[str]]] = {}
    for record in records:
        speaker_id = normalize_text_value(first_present(record, FIELD_CANDIDATES["speaker_id"]))
        if not speaker_id:
            continue
        attrs = attrs_by_speaker.setdefault(speaker_id, {})
        for attr in ATTRIBUTES:
            value = normalize_label(attr, first_present(record, FIELD_CANDIDATES[attr]))
            if value and not attrs.get(attr):
                attrs[attr] = value
    return attrs_by_speaker


def audio_keys(path: str) -> List[str]:
    p = Path(path)
    keys = [path, p.name, p.stem]
    if p.suffix.lower() == ".mp3":
        keys.extend([p.with_suffix(".wav").name, p.with_suffix(".wav").stem])
    elif p.suffix.lower() == ".wav":
        keys.extend([p.with_suffix(".mp3").name, p.with_suffix(".mp3").stem])
    return list(dict.fromkeys(keys))


def resolve_audio_path(value: Any, audio_root: Optional[Path]) -> Optional[str]:
    text = normalize_text_value(value)
    if text is None:
        return None
    path = Path(text)
    if path.is_absolute() or audio_root is None:
        return str(path)
    return str(audio_root / path)


def has_audio_hint(record: Dict[str, Any]) -> bool:
    if first_present(record, FIELD_CANDIDATES["audio_path"]) is not None:
        return True
    audios = record.get("audios")
    return isinstance(audios, list) and bool(audios)


def apply_audio_replacements(audio_path: str, replacements: Sequence[Tuple[str, str]]) -> str:
    for old, new in replacements:
        if audio_path.startswith(old):
            return new + audio_path[len(old) :]
    return audio_path


def lhotse_manifest_records(
    manifest_path: Optional[Path],
    replacements: Sequence[Tuple[str, str]],
) -> Iterable[Dict[str, Any]]:
    if manifest_path is None:
        return []
    try:
        from lhotse import load_manifest_lazy
    except ImportError as exc:
        raise RuntimeError(
            "Lhotse is required for --manifest_path but is not installed in this environment."
        ) from exc

    def _iter() -> Iterable[Dict[str, Any]]:
        cuts = load_manifest_lazy(str(manifest_path))
        for cut in cuts:
            if not getattr(cut, "supervisions", None):
                continue
            supervision = cut.supervisions[0]
            source = cut.recording.sources[0].source
            source = apply_audio_replacements(str(source), replacements)
            record = {
                "audio_path": source,
                "transcript": getattr(supervision, "text", None),
                "duration": getattr(cut, "duration", None),
                "speaker_id": getattr(supervision, "speaker", None),
            }
            custom = getattr(supervision, "custom", None)
            if isinstance(custom, dict):
                for key in FIELD_CANDIDATES["speaker_id"]:
                    if not record["speaker_id"] and key in custom:
                        record["speaker_id"] = custom[key]
                for attr in ATTRIBUTES:
                    for key in FIELD_CANDIDATES[attr]:
                        if key in custom:
                            record[attr] = custom[key]
                            break
            yield record

    return _iter()


def source_from_record(
    record: Dict[str, Any],
    salmonn_index: Dict[str, Dict[str, Any]],
    attrs_by_speaker: Dict[str, Dict[str, Optional[str]]],
    audio_root: Optional[Path],
    check_files: bool,
    skip_counts: Dict[str, int],
) -> Optional[SourceUtterance]:
    speaker_id = normalize_text_value(first_present(record, FIELD_CANDIDATES["speaker_id"]))
    raw_audio = first_present(record, FIELD_CANDIDATES["audio_path"])
    audio_path = resolve_audio_path(raw_audio, audio_root)
    transcript = normalize_text_value(first_present(record, FIELD_CANDIDATES["transcript"]))
    duration = as_float(first_present(record, FIELD_CANDIDATES["duration"]))

    audios = record.get("audios")
    if audio_path is None and isinstance(audios, list) and audios:
        audio_path = resolve_audio_path(audios[0], audio_root)
    messages_transcript = extract_assistant_text(record.get("messages"))
    transcript = transcript or messages_transcript
    durations = record.get("durations")
    if duration is None and isinstance(durations, list) and durations:
        duration = as_float(durations[0])

    joined = None
    if audio_path:
        for key in audio_keys(audio_path):
            joined = salmonn_index.get(key)
            if joined:
                break
    elif raw_audio is not None:
        for key in audio_keys(str(raw_audio)):
            joined = salmonn_index.get(key)
            if joined:
                break
    if joined:
        audio_path = audio_path or joined.get("audio_path")
        transcript = transcript or joined.get("transcript")
        duration = duration or joined.get("duration")

    speaker_attrs = attrs_by_speaker.get(speaker_id or "", {})
    gender = normalize_label("gender", first_present(record, FIELD_CANDIDATES["gender"])) or speaker_attrs.get("gender")
    age = normalize_label("age", first_present(record, FIELD_CANDIDATES["age"])) or speaker_attrs.get("age")
    accent = normalize_label("accent", first_present(record, FIELD_CANDIDATES["accent"])) or speaker_attrs.get("accent")

    if not speaker_id:
        skip_counts["missing_speaker_id"] += 1
        return None
    if not audio_path:
        skip_counts["missing_audio_path"] += 1
        return None
    if not transcript:
        skip_counts["missing_transcript"] += 1
        return None
    if not any((gender, age, accent)):
        skip_counts["missing_all_attributes"] += 1
        return None
    if check_files and not Path(audio_path).exists():
        skip_counts["missing_audio_file"] += 1
        return None

    return SourceUtterance(
        audio_path=str(audio_path),
        speaker_id=speaker_id,
        transcript=transcript,
        duration=duration,
        gender=gender,
        age=age,
        accent=accent,
    )


def load_sources(
    metadata_json: Path,
    input_json: Optional[Path],
    manifest_path: Optional[Path],
    audio_root: Optional[Path],
    replacements: Sequence[Tuple[str, str]],
    check_files: bool,
) -> Tuple[List[SourceUtterance], Dict[str, int]]:
    raw_metadata_records = unwrap_records(read_json(metadata_json))
    metadata_records = expand_metadata_records(raw_metadata_records)
    attrs_by_speaker = build_attr_by_speaker(raw_metadata_records + metadata_records)
    salmonn_index = build_salmonn_index(input_json)
    salmonn_records = load_salmonn_records(input_json)

    skip_counts: Dict[str, int] = collections.Counter()
    source_map: Dict[Tuple[str, str, str], SourceUtterance] = {}

    candidate_records: List[Dict[str, Any]] = []
    candidate_records.extend(record for record in metadata_records if has_audio_hint(record))
    candidate_records.extend(salmonn_records)
    candidate_records.extend(lhotse_manifest_records(manifest_path, replacements))

    for record in candidate_records:
        source = source_from_record(
            record=record,
            salmonn_index=salmonn_index,
            attrs_by_speaker=attrs_by_speaker,
            audio_root=audio_root,
            check_files=check_files,
            skip_counts=skip_counts,
        )
        if source is None:
            continue
        source_map[(source.audio_path, source.speaker_id, source.transcript)] = source

    return list(source_map.values()), dict(skip_counts)


def load_audio(path: str, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        gcd = math.gcd(sr, sample_rate)
        audio = resample_poly(audio, sample_rate // gcd, sr // gcd)
    return audio.astype(np.float32)


def save_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def audio_duration(path: str) -> Optional[float]:
    try:
        info = sf.info(path)
    except Exception:
        return None
    return float(info.frames) / float(info.samplerate) if info.samplerate else None


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio)) + EPS))


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > EPS:
        return (audio / peak * 0.9).astype(np.float32)
    return audio.astype(np.float32)


def sample_unique_speakers(
    rng: random.Random,
    pool: Sequence[SourceUtterance],
    n_sources: int,
    max_attempts: int = 200,
) -> List[SourceUtterance]:
    if len(pool) < n_sources:
        raise ValueError(f"Pool has {len(pool)} entries, need {n_sources}.")
    for _ in range(max_attempts):
        sampled = rng.sample(list(pool), n_sources)
        if len({src.speaker_id for src in sampled}) == n_sources:
            return sampled
    raise ValueError("Could not sample enough unique speaker IDs.")


def build_mixture_audio(
    mix_id: str,
    sources: Sequence[SourceUtterance],
    mixture_style: str,
    audio_out_dir: Path,
    sample_rate: int,
    rng: random.Random,
    max_delay_seconds: float,
    silence_range: Tuple[float, float],
    snr_range_db: Tuple[float, float],
) -> Dict[str, Any]:
    waves = [load_audio(src.audio_path, sample_rate) for src in sources]
    gains = [1.0 for _ in waves]

    if mixture_style == "sequential":
        segments = []
        cursor = 0
        total_parts: List[np.ndarray] = []
        for idx, wave in enumerate(waves):
            start = cursor
            total_parts.append(wave)
            cursor += len(wave)
            end = cursor
            if idx != len(waves) - 1:
                silence_seconds = rng.uniform(*silence_range)
                silence = np.zeros(int(round(silence_seconds * sample_rate)), dtype=np.float32)
                total_parts.append(silence)
                cursor += len(silence)
            segments.append((start, end))
        mixed = np.concatenate(total_parts) if total_parts else np.zeros(0, dtype=np.float32)
    elif mixture_style == "overlapped":
        ref_rms = rms(waves[0])
        for idx in range(1, len(waves)):
            snr_db = rng.uniform(*snr_range_db)
            target_rms = ref_rms * (10 ** (snr_db / 20.0))
            gains[idx] = target_rms / rms(waves[idx])
        waves = [wave * gain for wave, gain in zip(waves, gains)]
        starts = [0]
        for _ in waves[1:]:
            starts.append(int(round(rng.uniform(0.0, max_delay_seconds) * sample_rate)))
        ends = [start + len(wave) for start, wave in zip(starts, waves)]
        mixed = np.zeros(max(ends), dtype=np.float32)
        for wave, start, end in zip(waves, starts, ends):
            mixed[start:end] += wave
        segments = list(zip(starts, ends))
    else:
        raise ValueError(f"Unsupported mixture_style: {mixture_style}")

    mixed = normalize_peak(mixed)
    mix_path = audio_out_dir / f"{mix_id}.wav"
    save_audio(mix_path, mixed, sample_rate)

    source_records = []
    for index, (src, (start, end), gain) in enumerate(zip(sources, segments, gains), start=1):
        source_records.append(
            {
                "source_index": index,
                "speaker_id": src.speaker_id,
                "audio_path": src.audio_path,
                "transcript": src.transcript,
                "start_time": round(start / sample_rate, 6),
                "end_time": round(end / sample_rate, 6),
                "gain": round(float(gain), 6),
                "gender": src.gender,
                "age": src.age,
                "accent": src.accent,
            }
        )

    return {
        "mixture_id": mix_id,
        "mixture_path": str(mix_path),
        "mixture_duration": round(len(mixed) / sample_rate, 6),
        "mixture_style": mixture_style,
        "n_sources": len(sources),
        "sources": source_records,
    }


def single_audio_record(mix_id: str, source: SourceUtterance) -> Dict[str, Any]:
    duration = source.duration or audio_duration(source.audio_path)
    if duration is None:
        raise ValueError(f"Cannot determine duration for {source.audio_path}")
    return {
        "mixture_id": mix_id,
        "mixture_path": source.audio_path,
        "mixture_duration": round(duration, 6),
        "mixture_style": "single",
        "n_sources": 1,
        "sources": [
            {
                "source_index": 1,
                "speaker_id": source.speaker_id,
                "audio_path": source.audio_path,
                "transcript": source.transcript,
                "start_time": 0.0,
                "end_time": round(duration, 6),
                "gain": 1.0,
                "gender": source.gender,
                "age": source.age,
                "accent": source.accent,
            }
        ],
    }


def weighted_task_sequence(num_examples: int, rng: random.Random) -> List[str]:
    counts = []
    used = 0
    for idx, (name, weight) in enumerate(QUESTION_TYPE_WEIGHTS):
        if idx == len(QUESTION_TYPE_WEIGHTS) - 1:
            count = num_examples - used
        else:
            count = int(round(num_examples * weight))
            used += count
        counts.append((name, count))
    tasks: List[str] = []
    for name, count in counts:
        tasks.extend([name] * count)
    rng.shuffle(tasks)
    return tasks[:num_examples]


def choose_attribute(rng: random.Random, allowed: Sequence[str] = ATTRIBUTES) -> str:
    return rng.choice(list(allowed))


def label_universe(sources: Sequence[SourceUtterance]) -> Dict[str, List[str]]:
    labels: Dict[str, set] = {attr: set() for attr in ATTRIBUTES}
    for source in sources:
        for attr in ATTRIBUTES:
            label = source.label(attr)
            if label:
                labels[attr].add(label)
    return {attr: sorted(values) for attr, values in labels.items()}


def make_question_single(attribute: str) -> str:
    if attribute == "gender":
        return "What is the speaker's gender?"
    if attribute == "age":
        return "What is the speaker's age group?"
    return "What is the speaker's accent?"


def make_question_count(target_gender: str) -> str:
    return f"How many {target_gender} speakers are there in this audio?"


def make_question_ordinal(ordinal: str, attribute: str) -> str:
    if attribute == "gender":
        label_name = "gender"
    elif attribute == "age":
        label_name = "age group"
    else:
        label_name = "accent"
    return f"What is the {label_name} of the {ordinal} speaker?"


def make_question_keyword(keyword: str, attribute: str) -> str:
    if attribute == "gender":
        label_name = "gender"
    elif attribute == "age":
        label_name = "age group"
    else:
        label_name = "accent"
    return f"What is the {label_name} of the speaker who says \"{keyword}\"?"


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", text)


def find_unique_keyword(
    rng: random.Random,
    target_transcript: str,
    other_transcripts: Sequence[str],
) -> Optional[str]:
    words = tokenize_words(target_transcript)
    candidates = []
    for size in (4, 3, 2, 1):
        for start in range(0, max(0, len(words) - size + 1)):
            phrase = " ".join(words[start : start + size]).strip()
            if len(phrase) < 3:
                continue
            if size == 1 and len(phrase) < 5:
                continue
            lower_phrase = phrase.lower()
            if all(lower_phrase not in other.lower() for other in other_transcripts):
                candidates.append(phrase)
    if not candidates:
        return None
    return rng.choice(candidates)


def source_order(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    return sorted(record["sources"], key=lambda src: (src["start_time"], src["source_index"]))


def correct_source_for_ordinal(record: Dict[str, Any], ordinal: str) -> Dict[str, Any]:
    ordered = source_order(record)
    if ordinal == "first":
        return ordered[0]
    if ordinal == "second":
        return ordered[1]
    if ordinal == "last":
        return ordered[-1]
    raise ValueError(f"Unsupported ordinal: {ordinal}")


def qa_entry(
    record: Dict[str, Any],
    question: str,
    answer: str,
    question_type: str,
    attribute: str,
) -> Dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": f"<audio>{question}"},
            {"role": "assistant", "content": str(answer)},
        ],
        "audios": [record["mixture_path"]],
        "durations": [record["mixture_duration"]],
        "task_type": "speaker_analysis_qa",
        "question_type": question_type,
        "attribute": attribute,
        "answer": str(answer),
        "mixture_id": record["mixture_id"],
    }


def build_mc_question_text(question: str, options: Sequence[Tuple[str, str]]) -> str:
    option_lines = [f"Option {label}: {content}" for label, content in options]
    return (
        "<audio>Answer the following multiple-choice question using only the correct option.\n"
        f"Question: {question}\n"
        "Choices:\n"
        + "\n".join(option_lines)
        + "\n"
        + MC_FINAL_INSTRUCTION
    )


def mcq_entry(
    record: Dict[str, Any],
    question: str,
    correct_answer: str,
    options: Sequence[str],
    rng: random.Random,
) -> Dict[str, Any]:
    unique_options = list(dict.fromkeys(str(option) for option in options))
    if str(correct_answer) not in unique_options:
        unique_options.append(str(correct_answer))
    rng.shuffle(unique_options)
    labeled = [(LETTERS[idx], content) for idx, content in enumerate(unique_options)]
    answer_label = next(label for label, content in labeled if content == str(correct_answer))
    mc_options = [
        {
            "label": label,
            "content": content,
            "is_correct": label == answer_label,
        }
        for label, content in labeled
    ]
    return {
        "messages": [
            {"role": "user", "content": build_mc_question_text(question, labeled)},
            {"role": "assistant", "content": answer_label},
        ],
        "audios": [record["mixture_path"]],
        "task_type": "qa_mc",
        "mc_question": question,
        "mc_options": mc_options,
        "mc_answer_label": answer_label,
        "durations": [record["mixture_duration"]],
    }


def attribute_options(
    attribute: str,
    correct_answer: str,
    labels: Dict[str, List[str]],
    rng: random.Random,
    max_options: int,
) -> Optional[List[str]]:
    universe = [label for label in labels[attribute] if label != correct_answer]
    if not universe:
        return None
    desired = min(max_options, len(universe) + 1)
    distractors = rng.sample(universe, desired - 1)
    return [correct_answer] + distractors


def count_options(n_sources: int) -> List[str]:
    return [str(value) for value in range(n_sources + 1)]


def build_one_spec(
    task_type: str,
    idx: int,
    rng: random.Random,
    pools: Dict[str, List[SourceUtterance]],
    labels: Dict[str, List[str]],
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    mix_id = f"cv17_speaker_analysis_{idx:08d}"
    if task_type == "single_attribute":
        attribute = choose_attribute(rng)
        source = rng.choice(pools[attribute])
        answer = source.label(attribute)
        if not answer:
            raise ValueError("Sampled source missing attribute.")
        record = single_audio_record(mix_id, source)
        question = make_question_single(attribute)
        options = attribute_options(attribute, answer, labels, rng, args.max_mc_options)
        if not options:
            raise ValueError(f"Not enough labels for {attribute} MCQ.")
    else:
        mixture_style = "sequential" if rng.random() < args.sequential_ratio else "overlapped"
        if task_type == "gender_count":
            n_sources = rng.randint(args.min_mix_speakers, args.max_sequential_speakers if mixture_style == "sequential" else args.max_overlap_speakers)
            sampled = sample_unique_speakers(rng, pools["gender"], n_sources)
            target_gender = rng.choice(["female", "male"])
            question = make_question_count(target_gender)
            answer = str(sum(1 for source in sampled if source.gender == target_gender))
            attribute = "gender"
            options = count_options(n_sources)
        else:
            attribute = choose_attribute(rng)
            min_speakers = 2
            max_speakers = args.max_sequential_speakers if mixture_style == "sequential" else args.max_overlap_speakers
            n_sources = rng.randint(min_speakers, max_speakers)
            sampled = sample_unique_speakers(rng, pools[attribute], n_sources)
            options = None

        record = build_mixture_audio(
            mix_id=mix_id,
            sources=sampled,
            mixture_style=mixture_style,
            audio_out_dir=args.audio_out_dir,
            sample_rate=args.sample_rate,
            rng=rng,
            max_delay_seconds=args.max_delay_seconds,
            silence_range=(args.min_silence_seconds, args.max_silence_seconds),
            snr_range_db=(args.min_snr_db, args.max_snr_db),
        )

        if task_type == "ordinal_attribute":
            valid_ordinals = ["first", "last"]
            if record["n_sources"] >= 2:
                valid_ordinals.append("second")
            ordinal = rng.choice(valid_ordinals)
            target = correct_source_for_ordinal(record, ordinal)
            answer = target.get(attribute)
            if not answer:
                raise ValueError("Target source missing ordinal attribute.")
            question = make_question_ordinal(ordinal, attribute)
            options = attribute_options(attribute, answer, labels, rng, args.max_mc_options)
            if not options:
                raise ValueError(f"Not enough labels for {attribute} MCQ.")
        elif task_type == "keyword_attribute":
            ordered = source_order(record)
            target = rng.choice(ordered)
            others = [src["transcript"] for src in ordered if src is not target]
            keyword = find_unique_keyword(rng, target["transcript"], others)
            if keyword is None:
                raise ValueError("Could not find unique keyword.")
            answer = target.get(attribute)
            if not answer:
                raise ValueError("Target source missing keyword attribute.")
            question = make_question_keyword(keyword, attribute)
            options = attribute_options(attribute, answer, labels, rng, args.max_mc_options)
            if not options:
                raise ValueError(f"Not enough labels for {attribute} MCQ.")
            record["keyword"] = keyword
            record["keyword_source_index"] = target["source_index"]

    qa = qa_entry(record, question, str(answer), task_type, attribute)
    mcq = mcq_entry(record, question, str(answer), options, rng)
    record["question_type"] = task_type
    record["attribute"] = attribute
    record["question"] = question
    record["answer"] = str(answer)
    return record, qa, mcq


def validate_datasets(
    qa_entries: Sequence[Dict[str, Any]],
    mcq_entries: Sequence[Dict[str, Any]],
    metadata: Sequence[Dict[str, Any]],
    check_audio: bool,
) -> List[str]:
    issues = []
    expected_mcq_keys = {
        "messages",
        "audios",
        "task_type",
        "mc_question",
        "mc_options",
        "mc_answer_label",
        "durations",
    }
    for idx, entry in enumerate(qa_entries):
        if set(entry) < {"messages", "audios", "durations", "task_type"}:
            issues.append(f"qa[{idx}] missing required keys")
        if entry["task_type"] != "speaker_analysis_qa":
            issues.append(f"qa[{idx}] wrong task_type")
        if entry["messages"][0]["content"].count("<audio>") != len(entry["audios"]):
            issues.append(f"qa[{idx}] audio placeholder mismatch")
    for idx, entry in enumerate(mcq_entries):
        if set(entry.keys()) != expected_mcq_keys:
            issues.append(f"mcq[{idx}] keys do not match gemini_mc schema: {sorted(entry.keys())}")
        if entry.get("task_type") != "qa_mc":
            issues.append(f"mcq[{idx}] wrong task_type")
        correct = [option for option in entry["mc_options"] if option.get("is_correct") is True]
        if len(correct) != 1:
            issues.append(f"mcq[{idx}] does not have exactly one correct option")
        elif correct[0]["label"] != entry.get("mc_answer_label"):
            issues.append(f"mcq[{idx}] mc_answer_label mismatch")
        if entry["messages"][1]["content"] != entry.get("mc_answer_label"):
            issues.append(f"mcq[{idx}] assistant answer mismatch")
        if "mc_answer" in entry:
            issues.append(f"mcq[{idx}] contains forbidden mc_answer")
        if entry["messages"][0]["content"].count("<audio>") != len(entry["audios"]):
            issues.append(f"mcq[{idx}] audio placeholder mismatch")
    for idx, record in enumerate(metadata):
        speaker_ids = [src["speaker_id"] for src in record.get("sources", [])]
        if len(speaker_ids) != len(set(speaker_ids)):
            issues.append(f"metadata[{idx}] repeats speaker id")
        if record.get("question_type") == "keyword_attribute":
            keyword = record.get("keyword")
            if not keyword:
                issues.append(f"metadata[{idx}] missing keyword")
            else:
                matches = [
                    src
                    for src in record["sources"]
                    if keyword.lower() in src.get("transcript", "").lower()
                ]
                if len(matches) != 1:
                    issues.append(f"metadata[{idx}] keyword maps to {len(matches)} sources")
        if check_audio:
            path = record.get("mixture_path")
            actual_duration = audio_duration(path) if path else None
            expected_duration = record.get("mixture_duration")
            if actual_duration is None:
                issues.append(f"metadata[{idx}] audio is unreadable: {path}")
            elif abs(float(actual_duration) - float(expected_duration)) > 0.05:
                issues.append(f"metadata[{idx}] duration mismatch: {path}")
    return issues


def summarize(
    qa_entries: Sequence[Dict[str, Any]],
    metadata: Sequence[Dict[str, Any]],
    initial_skips: Dict[str, int],
    generation_skips: Dict[str, int],
    validation_issues: Sequence[str],
) -> Dict[str, Any]:
    return {
        "num_qa_entries": len(qa_entries),
        "num_mcq_entries": len(qa_entries),
        "question_type_counts": dict(collections.Counter(item["question_type"] for item in qa_entries)),
        "attribute_counts": dict(collections.Counter(item["attribute"] for item in qa_entries)),
        "answer_counts": dict(collections.Counter(item["answer"] for item in qa_entries)),
        "mixture_style_counts": dict(collections.Counter(record["mixture_style"] for record in metadata)),
        "n_sources_counts": dict(collections.Counter(str(record["n_sources"]) for record in metadata)),
        "initial_skip_counts": initial_skips,
        "generation_skip_counts": dict(generation_skips),
        "validation_issue_count": len(validation_issues),
        "validation_issues": list(validation_issues[:50]),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata_json", type=Path, required=True, help="Common Voice speaker metadata JSON.")
    parser.add_argument("--input_json", type=Path, default=None, help="Optional SALMONN ASR JSON to fill audio/transcript fields.")
    parser.add_argument("--manifest_path", type=Path, default=None, help="Optional Common Voice Lhotse manifest.")
    parser.add_argument("--audio_root", type=Path, default=None, help="Root used for relative audio paths in metadata.")
    parser.add_argument(
        "--replace_audio_prefix",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        default=[],
        help="Prefix rewrite for manifest audio paths. Can be repeated.",
    )
    parser.add_argument("--qa_output", type=Path, default=DEFAULT_QA_OUT)
    parser.add_argument("--mcq_output", type=Path, default=DEFAULT_MCQ_OUT)
    parser.add_argument("--metadata_output", type=Path, default=DEFAULT_METADATA_OUT)
    parser.add_argument("--report_output", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--audio_out_dir", type=Path, default=DEFAULT_OUT_DIR / "wav")
    parser.add_argument("--num_examples", type=int, default=25000, help="Examples per output file.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--min_mix_speakers", type=int, default=2)
    parser.add_argument("--max_sequential_speakers", type=int, default=4)
    parser.add_argument("--max_overlap_speakers", type=int, default=3)
    parser.add_argument("--sequential_ratio", type=float, default=0.7)
    parser.add_argument("--min_silence_seconds", type=float, default=0.2)
    parser.add_argument("--max_silence_seconds", type=float, default=0.8)
    parser.add_argument("--max_delay_seconds", type=float, default=2.0)
    parser.add_argument("--min_snr_db", type=float, default=-5.0)
    parser.add_argument("--max_snr_db", type=float, default=5.0)
    parser.add_argument("--max_mc_options", type=int, default=4)
    parser.add_argument("--max_retries_per_example", type=int, default=100)
    parser.add_argument("--no_check_input_files", action="store_true", help="Do not require source audio files to exist during source loading.")
    parser.add_argument("--skip_audio_validation", action="store_true", help="Skip output audio duration/readability validation.")
    return parser.parse_args()


def check_args(args: argparse.Namespace) -> None:
    if args.num_examples <= 0:
        raise ValueError("--num_examples must be positive.")
    if not 0.0 <= args.sequential_ratio <= 1.0:
        raise ValueError("--sequential_ratio must be between 0 and 1.")
    if args.min_mix_speakers < 2:
        raise ValueError("--min_mix_speakers must be at least 2.")
    if args.max_sequential_speakers < args.min_mix_speakers:
        raise ValueError("--max_sequential_speakers must be >= --min_mix_speakers.")
    if args.max_overlap_speakers < args.min_mix_speakers:
        raise ValueError("--max_overlap_speakers must be >= --min_mix_speakers.")
    if args.max_mc_options < 2 or args.max_mc_options > len(LETTERS):
        raise ValueError("--max_mc_options must be between 2 and 26.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    check_args(args)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    sources, initial_skips = load_sources(
        metadata_json=args.metadata_json,
        input_json=args.input_json,
        manifest_path=args.manifest_path,
        audio_root=args.audio_root,
        replacements=args.replace_audio_prefix,
        check_files=not args.no_check_input_files,
    )
    if not sources:
        raise RuntimeError("No usable Common Voice sources after filtering.")

    labels = label_universe(sources)
    pools = {attr: [source for source in sources if source.label(attr)] for attr in ATTRIBUTES}
    for attr in ATTRIBUTES:
        if len({src.speaker_id for src in pools[attr]}) < args.min_mix_speakers:
            raise RuntimeError(f"Not enough unique speakers with {attr} labels.")
        if len(labels[attr]) < 2:
            raise RuntimeError(f"Need at least two {attr} labels for MCQ generation.")

    logging.info("Loaded %d usable sources.", len(sources))
    logging.info("Label counts: %s", {attr: len(values) for attr, values in labels.items()})

    metadata_records: List[Dict[str, Any]] = []
    qa_entries: List[Dict[str, Any]] = []
    mcq_entries: List[Dict[str, Any]] = []
    generation_skips: Dict[str, int] = collections.Counter()
    tasks = weighted_task_sequence(args.num_examples, rng)

    for idx, task_type in enumerate(tqdm(tasks, desc="Generating")):
        last_error = None
        for _ in range(args.max_retries_per_example):
            try:
                record, qa, mcq = build_one_spec(task_type, idx, rng, pools, labels, args)
                metadata_records.append(record)
                qa_entries.append(qa)
                mcq_entries.append(mcq)
                break
            except Exception as exc:
                last_error = exc
                generation_skips[f"{task_type}:{type(exc).__name__}:{str(exc)[:80]}"] += 1
        else:
            raise RuntimeError(
                f"Failed to generate example {idx} for {task_type} after "
                f"{args.max_retries_per_example} retries. Last error: {last_error}"
            )

    validation_issues = validate_datasets(
        qa_entries=qa_entries,
        mcq_entries=mcq_entries,
        metadata=metadata_records,
        check_audio=not args.skip_audio_validation,
    )
    if validation_issues:
        for issue in validation_issues[:20]:
            logging.error("Validation issue: %s", issue)
        raise RuntimeError(f"Validation failed with {len(validation_issues)} issue(s).")

    report = summarize(qa_entries, metadata_records, initial_skips, generation_skips, validation_issues)
    write_json(args.metadata_output, metadata_records)
    write_json(args.qa_output, {"data": qa_entries})
    write_json(args.mcq_output, {"data": mcq_entries})
    write_json(args.report_output, report)

    logging.info("Wrote QA JSON to %s", args.qa_output)
    logging.info("Wrote MCQ JSON to %s", args.mcq_output)
    logging.info("Wrote metadata to %s", args.metadata_output)
    logging.info("Wrote report to %s", args.report_output)


if __name__ == "__main__":
    main()
