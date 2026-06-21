#!/usr/bin/env python3
"""Generate Gemini-style analytical QA pairs from MusicBench metadata.

This pipeline reads MusicBench JSONL rows, keeps one original plus a small
number of augmentations per base clip, and writes SALMONN-style QA JSON:

{"data": [
{"messages": [...], "audios": [...], "durations": [...], "task_type": "QA"},
...
]}
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

HELPER_DIR = Path(__file__).resolve().parents[1] / "music_qa_from_caption"
sys.path.insert(0, str(HELPER_DIR))

from qwen_music_qa_pipeline import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_MODEL_PATH,
    append_failed_record,
    call_local_qwen,
    call_local_qwen_batch,
    call_qwen,
    load_checkpoint,
    load_local_qwen,
    strip_markdown_fence,
    write_compact_salmonn_json,
)


DEFAULT_METADATA = "MusicBench_train.json"
DEFAULT_OUTPUT = "salmonn_data_v1.1/musicbench/musicbench_train_qwen3.5_35b_a3b_template_qa.json"
DEFAULT_NUM_QA = 2
DEFAULT_AUGMENTATIONS_PER_BASE = 1
DEFAULT_DURATION = 10.0

LEAKAGE_PATTERNS = [
    r"\bcaption\b",
    r"\bdescription\b",
    r"\btext description\b",
    r"\bprovided text\b",
    r"\bprovided information\b",
    r"\bsource text\b",
    r"\bprompt\b",
    r"\bmetadata\b",
    r"\bmentions?\b",
    r"\bmentioned\b",
    r"\bas stated\b",
    r"\bthe text\b",
    r"\bthe source\b",
]

BAD_GENERATION_PATTERNS = [
    r"\bexplicitly (?:described|stated|mentioned)\b",
    r"\blikely captured\b",
    r"\bcaptured in a single take\b",
    r"\bsingle[- ]take\b",
    r"\brequires a clear\b",
    r"\bnatural reverb tail\b",
    r"\bconfirm(?:s|ed)? that\b",
]

SOURCE_CONDITIONAL_PATTERNS = [
    (r"\breverb(?:erant| tail)?\b|\breverberant\b|\becho(?:ey|ing)?\b", r"\breverb|\breverber|\becho"),
    (r"\bstudio\b", r"\bstudio\b"),
    (r"\bmicrophone\b|\bmic\b", r"\bmicrophone\b|\bmic\b"),
    (r"\bsingle[- ]take\b|\bone take\b", r"\bsingle[- ]take\b|\bone take\b"),
    (r"\btriplet feel\b|\btriplets?\b", r"\btriplets?\b"),
    (r"\bdry acoustic space\b|\bdry room\b|\bdry production\b|\bdry mix\b", r"\bdry\b"),
]


META_QA_TEMPLATES = [
    {
        "id": "holistic_style_mood",
        "instruction": "Generate one QA pair that asks what overall style, mood, or expressive character can be inferred from the audio, using multiple musical cues as evidence.",
        "patterns": [r"\b(style|genre|mood|emotion|atmosphere|character|feeling|expressive|relaxing|energetic|melancholic|joyful|tense)\b"],
    },
    {
        "id": "contrastive_classification",
        "instruction": "Generate one QA pair that asks why the music is better described as one style, mood, or category rather than another plausible but incorrect one. Rule out the incorrect alternative using only audible evidence present in the description.",
        "patterns": [r"\b(style|genre|mood|atmosphere|character|instrumental|song|track)\b"],
    },
    {
        "id": "instrument_timbre_expression",
        "instruction": "Generate one QA pair about how a specific instrument's timbre, articulation, or playing role contributes to the music's emotional or stylistic effect.",
        "patterns": [
            r"\b(piano|guitar|drum|bass|violin|viola|cello|flute|saxophone|trumpet|clarinet|strings?|synthesizer|synth|vocal|voice|percussion|instrument)\b",
            r"\b(tone|timbre|plays?|playing|lead|main|dominant|bright|warm|mellow|crisp|distorted|soft|heavy|plucked|strummed|arpeggiated)\b",
        ],
    },
    {
        "id": "rhythm_dynamics_energy",
        "instruction": "Generate one QA pair focused on rhythm, tempo, dynamics, groove, pulse, or forward momentum, and explain how these features shape the music's energy.",
        "patterns": [r"\b(rhythm|tempo|beat|pulse|groove|meter|bpm|pace|dynamic|dynamics|volume|energy|energetic|momentum|driving|steady|fast|slow|moderate|4/4|common time)\b"],
    },
    {
        "id": "harmony_tonality_structure",
        "instruction": "Generate one QA pair about harmony, key, chord progression, tonal center, or meter, but only if those details are explicitly supported. The answer should explain how these structural details affect the perceived character of the music.",
        "patterns": [r"\b(key|major|minor|chord|progression|tonal|harmony|harmonic|meter|4/4|common time|beat counts?)\b"],
    },
    {
        "id": "negative_evidence_rule_out",
        "instruction": "Generate one QA pair that uses an explicitly stated absence, such as no vocals, sparse instrumentation, no drums, or no dense arrangement, to rule out another interpretation.",
        "patterns": [r"\b(no |without|lacks?|absence|does not|do not|not feature|instrumental|no voices|no vocals|only|solo)\b"],
    },
    {
        "id": "scene_functional_suitability",
        "instruction": "Generate one QA pair that asks what kind of scene, setting, background use, or listening context the music would support, grounded strictly in musical evidence.",
        "patterns": [r"\b(suitable|perfect for|background|scene|setting|context|film|cinematic|soundtrack|game|dance|meditation|relaxation|coffee shop|advertisement|tv|show)\b"],
    },
    {
        "id": "temporal_structural_change",
        "instruction": "Generate one QA pair about changes over time, such as an opening gesture, later section, shift in texture, added instrument, dynamic change, or phrase development. Only ask about change that is explicitly stated.",
        "patterns": [r"\b(begins?|starts?|opens?|then|later|followed by|after|builds?|develops?|changes?|shift|section|crescendo|decrescendo|ending)\b"],
    },
    {
        "id": "production_recording_texture",
        "instruction": "Generate one QA pair about explicitly stated recording or production evidence, such as low sound quality, distortion, reverb, an empty-room sound, electronic treatment, or clear sonic texture. Restate weak recording clues conservatively and do not infer studio setup, microphone technique, single-take capture, room acoustics, or causes of the sound quality.",
        "patterns": [r"\b(reverb|distortion|recording|recorded|sound quality|low quality|poor quality|lo-fi|empty room|electronic treatment|sonic clarity|clear sound|unclear|noisy)\b"],
    },
    {
        "id": "augmentation_invariant_core",
        "instruction": "Generate one QA pair about musical qualities that remain central despite this being an augmented version, such as instrumentation, role separation, style, absence of vocals, or functional mood. Avoid implying the listener knows it is augmented.",
        "patterns": [r"\b(instrument|style|mood|rhythm|beat|chord|key|bpm|vocal|voice|song|track)\b"],
        "requires_augmented": True,
    },
]


PROMPT_TEMPLATE = """
Role: You are a world-class expert in music analysis, audio understanding, and dataset construction for audio-language models.

Task: You will be given a detailed natural-language music description and one specialized QA-generation focus. Based only on information explicitly contained in the description, generate exactly one high-quality audio/music-grounded question-answer pair.

Input:
A detailed music description:
{music_description}

Specialized QA focus:
{template_instruction}

Precision guidance:
{precision_instruction}

Requirements for the question:
1. The question must begin with "<audio>".
2. The question should sound as if it is being asked about the audio itself, not about a caption, text, prompt, or metadata record.
3. The question should be analytical and require synthesizing multiple musical details.
4. The question must stay within the specialized focus above.
5. Do not ask about unsupported facts such as composer, artist, microphone placement, exact chord names, exact key, or exact BPM unless those facts are explicitly included.
6. Vary the question starter naturally. Prefer evidence-seeking, contrastive, or inference-based openings such as "What evidence suggests...", "Why is this better characterized as...", "Which cues indicate...", "What makes...", or "What can be inferred from...".

Requirements for the answer:
1. The answer must be fully supported by the given music information.
2. The answer should cite multiple musical cues and explain how they support the conclusion.
3. The answer should be concise but substantive, usually 2-5 sentences.
4. Do not introduce new facts beyond the music information.
5. Do not mention captions, descriptions, prompts, metadata, provided text, or source text. Present all evidence as audible musical evidence.
6. Use cautious, evidence-grounded language. Avoid overclaiming with words such as "obviously", "definitively", or "undoubtedly" unless the evidence directly warrants it.
7. Do not infer production setup, live/studio context, reverb, room acoustics, microphone placement, single-take performance, or physical recording conditions unless those details are directly stated. For weak clues like "low quality" or "empty room", keep the explanation close to those words without inventing a cause.

Output format:
Return only valid JSON as one object:

{{
  "question": "<audio>...",
  "answer": "..."
}}
""".strip()


def current_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def get_rank_id() -> str:
    for key in ("RANK_ID", "RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value:
            return value
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return f"gpu:{cuda_visible_devices}"
    return "main"


def log_message(message: str, level: str = "INFO") -> None:
    print(f"[{current_timestamp()}][rank {get_rank_id()}][{level}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", "--input", dest="metadata", default=DEFAULT_METADATA, help="Input MusicBench JSONL metadata.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output SALMONN-style QA JSON.")
    parser.add_argument("--tmp-output", default="", help="Checkpoint JSONL path. Defaults to <output>.tmp.jsonl.")
    parser.add_argument("--failed-output", default="", help="Failed-sample JSONL path. Defaults to <output>.failed.jsonl.")
    parser.add_argument("--audio-root", default="", help="Root used to resolve relative MusicBench locations. Defaults to metadata parent.")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION, help="Duration stored in each output item.")
    parser.add_argument("--augmentations-per-base", type=int, default=DEFAULT_AUGMENTATIONS_PER_BASE, help="Number of augmented clips to keep per base clip.")
    parser.add_argument("--num-qa", type=int, default=DEFAULT_NUM_QA, help="Maximum number of applicable meta-templates sampled per selected clip.")
    parser.add_argument("--seed", type=int, default=20260621, help="Seed for deterministic row and template sampling.")
    parser.add_argument("--rank", type=int, default=0, help="Shard rank for multi-process generation.")
    parser.add_argument("--world-size", type=int, default=1, help="Number of shards for multi-process generation.")
    parser.add_argument("--selection-stats-only", action="store_true", help="Print selected clip counts and exit without loading a model.")

    parser.add_argument("--backend", choices=("local", "openai"), default="local", help="Run local transformers inference or use an OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key, preferably via OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name exposed by the endpoint.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path or Hugging Face model id.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum generated tokens for either backend.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True, help="Use sampling for local generation.")
    parser.add_argument("--torch-dtype", default="auto", choices=("auto", "bfloat16", "float16", "float32"), help="Local model dtype.")
    parser.add_argument("--device-map", default="auto", help="Local model device_map, e.g. auto, cuda:0, cpu.")
    parser.add_argument("--attn-implementation", default="", help="Optional local attention implementation, e.g. flash_attention_2 or sdpa.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Pass trust_remote_code=True when loading the local model/tokenizer.")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--sample-batch-size", type=int, default=1, help="Number of source clips to accumulate before one batched local generation call.")
    parser.add_argument("--debug", action="store_true", help="Only process the first 5 selected clips after sharding.")
    parser.add_argument("--limit", type=int, default=0, help="Optional positive selected-clip limit after sharding. Overrides --debug.")
    parser.add_argument("--finalize-only", action="store_true", help="Only convert the checkpoint JSONL to final SALMONN JSON.")
    return parser.parse_args()


def stable_int(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


def load_musicbench_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_row_index"] = line_index
            rows.append(row)
    return rows


def base_location(location: str) -> str:
    loc = location.replace("\\", "/")
    if loc.startswith("data_aug2/"):
        loc = "data/" + loc[len("data_aug2/"):]
        loc = re.sub(r"_(\d+)(\.wav)$", r"\2", loc)
    return loc


def is_augmented(row: dict[str, Any]) -> bool:
    return str(row.get("location", "")).replace("\\", "/").startswith("data_aug2/")


def augmentation_kind(row: dict[str, Any], original: dict[str, Any] | None) -> str:
    if original is None:
        return "augmentation"
    row_bpm = row.get("bpm")
    orig_bpm = original.get("bpm")
    row_key = row.get("key")
    orig_key = original.get("key")
    bpm_changed = row_bpm is not None and orig_bpm is not None and abs(float(row_bpm) - float(orig_bpm)) > 1e-6
    key_changed = row_key is not None and orig_key is not None and row_key != orig_key
    if bpm_changed and not key_changed:
        return "tempo_augmentation"
    if key_changed and not bpm_changed:
        return "pitch_augmentation"
    if key_changed and bpm_changed:
        return "tempo_pitch_augmentation"
    return "augmentation"


def stable_sample(rows: list[dict[str, Any]], n: int, seed: int, salt: str) -> list[dict[str, Any]]:
    keyed = sorted(rows, key=lambda row: stable_int(f"{seed}\0{salt}\0{row.get('location', '')}\0{row.get('_row_index', 0)}"))
    return keyed[:n]


def select_musicbench_rows(rows: list[dict[str, Any]], augmentations_per_base: int, seed: int) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        grouped[base_location(str(row.get("location", "")))].append(row)

    selected: list[dict[str, Any]] = []
    for base in sorted(grouped):
        group = grouped[base]
        originals = [row for row in group if not is_augmented(row)]
        augmentations = [row for row in group if is_augmented(row)]

        original = stable_sample(originals, 1, seed, base + "\0original")[0] if originals else None
        if original is not None:
            selected.append(original)

        aug_quota = max(0, augmentations_per_base)
        if original is None and augmentations:
            aug_quota = max(1, aug_quota)
        if aug_quota == 0 or not augmentations:
            continue

        if aug_quota >= 2:
            by_kind: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for row in augmentations:
                by_kind[augmentation_kind(row, original)].append(row)

            chosen: list[dict[str, Any]] = []
            for kind in ("tempo_augmentation", "pitch_augmentation", "tempo_pitch_augmentation", "augmentation"):
                if len(chosen) >= aug_quota:
                    break
                candidates = [row for row in by_kind.get(kind, []) if row not in chosen]
                if candidates:
                    chosen.extend(stable_sample(candidates, 1, seed, base + "\0" + kind))

            remaining = [row for row in augmentations if row not in chosen]
            chosen.extend(stable_sample(remaining, aug_quota - len(chosen), seed, base + "\0remaining"))
            selected.extend(chosen[:aug_quota])
        else:
            selected.extend(stable_sample(augmentations, aug_quota, seed, base + "\0augmentation"))

    return selected


def audio_path(row: dict[str, Any], audio_root: Path) -> str:
    location = Path(str(row["location"]))
    if location.is_absolute():
        return str(location)
    return str(audio_root / location)


def key_text(row: dict[str, Any]) -> str:
    key = row.get("key")
    if isinstance(key, list) and len(key) >= 2:
        value = f"{key[0]} {key[1]}"
    elif key:
        value = str(key)
    else:
        return ""
    keyprob = row.get("keyprob")
    if isinstance(keyprob, list) and keyprob:
        return f"The detected key is {value} with confidence {float(keyprob[0]):.2f}."
    return f"The detected key is {value}."


def chord_text(row: dict[str, Any], max_chords: int = 8) -> str:
    chords = row.get("chords") or []
    if not chords:
        return ""
    unique: list[str] = []
    for chord in chords:
        if chord not in unique:
            unique.append(str(chord))
        if len(unique) >= max_chords:
            break
    return "The chord material includes " + ", ".join(unique) + "."


def build_music_description(row: dict[str, Any]) -> str:
    parts = []
    for key in ("main_caption", "alt_caption"):
        value = str(row.get(key, "")).strip()
        if value:
            parts.append(value)

    structured = []
    for key in ("prompt_bpm", "prompt_key", "prompt_ch", "prompt_bt"):
        value = str(row.get(key, "")).strip()
        if value:
            structured.append(value)
    if not any("key" in item.lower() for item in structured):
        value = key_text(row)
        if value:
            structured.append(value)
    if not any("chord" in item.lower() for item in structured):
        value = chord_text(row)
        if value:
            structured.append(value)

    if structured:
        parts.append("Additional musical facts: " + " ".join(structured))
    return "\n\n".join(parts).strip()


def build_source_item(row: dict[str, Any], source_index: int, audio_root: Path, duration: float) -> dict[str, Any]:
    return {
        "source_index": source_index,
        "global_row_index": row.get("_row_index"),
        "base_location": base_location(str(row["location"])),
        "location": row["location"],
        "is_augmented": is_augmented(row),
        "caption": build_music_description(row),
        "audios": [audio_path(row, audio_root)],
        "durations": [duration],
    }


def build_qa_item(source_item: dict[str, Any], qa: dict[str, str]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ],
        "audios": source_item["audios"],
        "durations": source_item["durations"],
        "task_type": "QA",
    }


def template_is_applicable(template: dict[str, Any], source_item: dict[str, Any]) -> bool:
    if template.get("requires_augmented") and not source_item["is_augmented"]:
        return False
    caption = source_item["caption"].lower()
    return all(re.search(pattern, caption) for pattern in template["patterns"])


def get_applicable_templates(source_item: dict[str, Any]) -> list[dict[str, Any]]:
    templates = [template for template in META_QA_TEMPLATES if template_is_applicable(template, source_item)]
    return templates or [META_QA_TEMPLATES[0]]


def sample_templates(source_item: dict[str, Any], source_index: int, num_templates: int, seed: int) -> list[dict[str, Any]]:
    templates = get_applicable_templates(source_item)
    if len(templates) <= num_templates:
        return templates
    rng = random.Random(seed + source_index)
    return rng.sample(templates, num_templates)


def precision_instruction(template: dict[str, Any]) -> str:
    if template["id"] in {"harmony_tonality_structure", "rhythm_dynamics_energy"}:
        return (
            "You may use exact BPM, meter, key, or chord labels when they are present, "
            "but do not make the QA a simple metadata lookup. Connect those facts to "
            "audible musical effect, such as energy, stability, mood, or structure."
        )
    return (
        "Avoid centering the QA on exact BPM, key, chord names, or other label-like facts. "
        "If such facts are present, translate them into broader musical effects and "
        "prioritize instrumentation, texture, rhythm, mood, and arrangement evidence."
    )


def build_template_prompt(music_description: str, template: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        music_description=music_description.strip(),
        template_instruction=template["instruction"],
        precision_instruction=precision_instruction(template),
    )


def find_leakage(text: str) -> str | None:
    for pattern in LEAKAGE_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return pattern
    return None


def find_quality_issue(text: str, source_text: str) -> str | None:
    source_text = source_text.lower()
    for pattern in BAD_GENERATION_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return f"over-inference or meta phrasing matching {pattern!r}"

    for output_pattern, source_pattern in SOURCE_CONDITIONAL_PATTERNS:
        if re.search(output_pattern, text, flags=re.IGNORECASE) and not re.search(source_pattern, source_text, flags=re.IGNORECASE):
            return f"unsupported production/detail inference matching {output_pattern!r}"
    return None


def parse_single_qa_response(text: str, source_text: str = "") -> dict[str, str]:
    text = strip_markdown_fence(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start_obj = text.find("{")
        end_obj = text.rfind("}")
        if start_obj == -1 or end_obj == -1 or end_obj <= start_obj:
            raise
        parsed = json.loads(text[start_obj : end_obj + 1])

    if isinstance(parsed, list):
        if len(parsed) != 1:
            raise ValueError(f"Expected one QA object, got a list of {len(parsed)}.")
        parsed = parsed[0]
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected one JSON object, got {type(parsed).__name__}.")

    question = str(parsed.get("question", "")).strip()
    answer = str(parsed.get("answer", "")).strip()
    if not question or not answer:
        raise ValueError("Generated QA has an empty question or answer.")
    if question.startswith("<audio>"):
        question = "<audio>" + question[len("<audio>"):].lstrip()
    else:
        question = "<audio>" + question.lstrip()

    leakage_pattern = find_leakage(f"{question}\n{answer}")
    if leakage_pattern:
        raise ValueError(f"Generated QA leaks meta-input wording matching {leakage_pattern!r}.")
    quality_issue = find_quality_issue(f"{question}\n{answer}", source_text)
    if quality_issue:
        raise ValueError(f"Generated QA failed quality filter: {quality_issue}.")
    return {"question": question, "answer": answer}


def finalize_output(tmp_path: Path, output_path: Path, source_count: int) -> tuple[int, int]:
    records = load_checkpoint(tmp_path)
    missing = [idx for idx in range(source_count) if idx not in records]
    output_items = []
    for idx in sorted(records):
        output_items.extend(records[idx]["qa_items"])
    write_compact_salmonn_json(output_path, output_items)
    return len(output_items), len(missing)


def load_generator(args: argparse.Namespace):
    if args.backend == "openai":
        if not args.base_url:
            raise ValueError("Missing --base-url for OpenAI-compatible backend.")
        if not args.api_key:
            raise ValueError("Missing --api-key for OpenAI-compatible backend.")
        from openai import OpenAI

        return OpenAI(base_url=args.base_url, api_key=args.api_key)
    return load_local_qwen(args)


def generate_one(generator, args: argparse.Namespace, prompt: str) -> str:
    if args.backend == "openai":
        return call_qwen(generator, args, prompt)
    model, tokenizer = generator
    return call_local_qwen(model, tokenizer, args, prompt)


def generate_many(generator, args: argparse.Namespace, prompts: list[str]) -> list[str]:
    if args.backend == "openai":
        return [call_qwen(generator, args, prompt) for prompt in prompts]
    model, tokenizer = generator
    return call_local_qwen_batch(model, tokenizer, args, prompts)


def build_source_job(source_item: dict[str, Any], source_index: int, args: argparse.Namespace) -> dict[str, Any]:
    templates = sample_templates(source_item, source_index, args.num_qa, args.seed)
    return {
        "source_index": source_index,
        "source_item": source_item,
        "selected_templates": templates,
        "prompts": [build_template_prompt(source_item["caption"], template) for template in templates],
    }


def generate_source_batch(generator, args: argparse.Namespace, jobs: list[dict[str, Any]]):
    prompt_entries = []
    for job in jobs:
        for template, prompt in zip(job["selected_templates"], job["prompts"]):
            prompt_entries.append({
                "source_index": job["source_index"],
                "template": template,
                "prompt": prompt,
                "source_text": job["source_item"]["caption"],
            })
    if not prompt_entries:
        return {}, []

    source_ids = ",".join(str(job["source_index"]) for job in jobs)
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_texts = generate_many(generator, args, [entry["prompt"] for entry in prompt_entries])
            if len(raw_texts) != len(prompt_entries):
                raise ValueError(f"Expected {len(prompt_entries)} batch outputs, got {len(raw_texts)}.")
            grouped_qas = {job["source_index"]: [] for job in jobs}
            for entry, raw_text in zip(prompt_entries, raw_texts):
                qa = parse_single_qa_response(raw_text, entry["source_text"])
                grouped_qas[entry["source_index"]].append((entry["template"], qa))
            return grouped_qas, []
        except Exception as exc:
            last_error = exc
            log_message(
                f"source_batch={source_ids} prompts={len(prompt_entries)} attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
                level="WARN",
            )
            print(traceback.format_exc(), flush=True)
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)

    errors = [
        {
            "source_index": entry["source_index"],
            "template_id": entry["template"]["id"],
            "error": repr(last_error),
            "source_batch_failed": True,
        }
        for entry in prompt_entries
    ]
    return None, errors


def process_one_source(generator, args: argparse.Namespace, job: dict[str, Any]):
    source_index = job["source_index"]
    source_item = job["source_item"]
    qa_items = []
    succeeded_template_ids = []
    template_errors = []

    for template, prompt in zip(job["selected_templates"], job["prompts"]):
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                qa = parse_single_qa_response(generate_one(generator, args, prompt), source_item["caption"])
                qa_items.append(build_qa_item(source_item, qa))
                succeeded_template_ids.append(template["id"])
                break
            except Exception as exc:
                last_error = exc
                log_message(
                    f"source={source_index} template={template['id']} attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
                    level="WARN",
                )
                print(traceback.format_exc(), flush=True)
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep * attempt)
        else:
            template_errors.append({"template_id": template["id"], "error": repr(last_error)})
    return qa_items, succeeded_template_ids, template_errors


def write_job_result(
    tmp_f,
    failed_path: Path,
    job: dict[str, Any],
    qa_items: list[dict[str, Any]],
    succeeded_template_ids: list[str],
    template_errors: list[dict[str, Any]],
    total_count: int,
) -> None:
    source_index = job["source_index"]
    selected_templates = job["selected_templates"]
    if qa_items:
        record = {
            "source_index": source_index,
            "source_location": job["source_item"]["location"],
            "base_location": job["source_item"]["base_location"],
            "is_augmented": job["source_item"]["is_augmented"],
            "template_ids": succeeded_template_ids,
            "qa_items": qa_items,
            "template_errors": template_errors,
        }
        tmp_f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp_f.flush()
        log_message(
            f"source={source_index} progress={source_index + 1}/{total_count} generated={len(qa_items)} "
            f"selected_templates={len(selected_templates)} template_ids={','.join(succeeded_template_ids)}"
        )
    else:
        append_failed_record(
            failed_path,
            {
                "source_index": source_index,
                "source_location": job["source_item"]["location"],
                "selected_template_ids": [template["id"] for template in selected_templates],
                "template_errors": template_errors,
            },
        )
        log_message(f"Skipping source index {source_index}; all selected templates failed. Logged to {failed_path}", level="ERROR")


def process_source_jobs(generator, args: argparse.Namespace, jobs: list[dict[str, Any]], tmp_f, failed_path: Path, total_count: int) -> None:
    grouped_qas, batch_errors = generate_source_batch(generator, args, jobs)
    batch_errors_by_source: dict[int, list[dict[str, Any]]] = {}
    for error in batch_errors:
        batch_errors_by_source.setdefault(error["source_index"], []).append(error)

    if grouped_qas is None:
        log_message(
            f"source_batch={','.join(str(job['source_index']) for job in jobs)} falling back to per-source generation",
            level="WARN",
        )
        for job in jobs:
            qa_items, succeeded_template_ids, template_errors = process_one_source(generator, args, job)
            template_errors = batch_errors_by_source.get(job["source_index"], []) + template_errors
            write_job_result(tmp_f, failed_path, job, qa_items, succeeded_template_ids, template_errors, total_count)
        return

    for job in jobs:
        source_item = job["source_item"]
        template_qa_pairs = grouped_qas[job["source_index"]]
        qa_items = [build_qa_item(source_item, qa) for _, qa in template_qa_pairs]
        succeeded_template_ids = [template["id"] for template, _ in template_qa_pairs]
        write_job_result(tmp_f, failed_path, job, qa_items, succeeded_template_ids, [], total_count)


def build_sources(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, int]]:
    metadata_path = Path(args.metadata)
    audio_root = Path(args.audio_root) if args.audio_root else metadata_path.parent
    rows = load_musicbench_rows(metadata_path)
    selected_rows = select_musicbench_rows(rows, args.augmentations_per_base, args.seed)
    sharded_rows = [
        row
        for selected_index, row in enumerate(selected_rows)
        if selected_index % args.world_size == args.rank
    ]

    limit = args.limit if args.limit > 0 else (5 if args.debug else len(sharded_rows))
    limit = min(limit, len(sharded_rows))
    sharded_rows = sharded_rows[:limit]

    sources = [
        build_source_item(row, source_index, audio_root, args.duration)
        for source_index, row in enumerate(sharded_rows)
    ]
    stats = {
        "input_rows": len(rows),
        "selected_rows_all_ranks": len(selected_rows),
        "selected_rows_this_rank": len(sources),
        "augmented_rows_this_rank": sum(1 for item in sources if item["is_augmented"]),
        "original_rows_this_rank": sum(1 for item in sources if not item["is_augmented"]),
        "estimated_qa_this_rank": sum(min(args.num_qa, len(get_applicable_templates(item))) for item in sources),
    }
    return sources, stats


def validate_args(args: argparse.Namespace) -> None:
    if args.augmentations_per_base < 0:
        raise ValueError("--augmentations-per-base must be non-negative.")
    if args.num_qa < 1:
        raise ValueError("--num-qa must be at least 1.")
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be at least 1.")
    if args.world_size < 1:
        raise ValueError("--world-size must be at least 1.")
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("--rank must satisfy 0 <= rank < world-size.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    output_path = Path(args.output)
    tmp_path = Path(args.tmp_output) if args.tmp_output else Path(str(output_path) + ".tmp.jsonl")
    failed_path = Path(args.failed_output) if args.failed_output else Path(str(output_path) + ".failed.jsonl")

    sources, stats = build_sources(args)
    log_message(
        "selection stats: "
        + ", ".join(f"{key}={value}" for key, value in stats.items())
        + f", rank={args.rank}, world_size={args.world_size}, qas_per_clip={args.num_qa}"
    )

    if args.selection_stats_only:
        return

    if args.finalize_only:
        total, missing = finalize_output(tmp_path, output_path, len(sources))
        log_message(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")
        return

    generator = load_generator(args)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(tmp_path)

    with tmp_path.open("a", encoding="utf-8") as tmp_f:
        pending_jobs = []
        for source_index, source_item in enumerate(sources):
            if source_index in checkpoint:
                continue
            pending_jobs.append(build_source_job(source_item, source_index, args))
            if len(pending_jobs) >= args.sample_batch_size:
                process_source_jobs(generator, args, pending_jobs, tmp_f, failed_path, len(sources))
                pending_jobs = []

        if pending_jobs:
            process_source_jobs(generator, args, pending_jobs, tmp_f, failed_path, len(sources))

    total, missing = finalize_output(tmp_path, output_path, len(sources))
    log_message(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")


if __name__ == "__main__":
    main()
