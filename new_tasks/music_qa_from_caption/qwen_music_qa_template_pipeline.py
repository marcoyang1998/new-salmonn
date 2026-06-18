import argparse
import json
import os
import random
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from qwen_music_qa_pipeline import (
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PATH,
    append_failed_record,
    call_local_qwen_batch,
    build_qa_salmonn_item,
    call_local_qwen,
    call_qwen,
    get_caption,
    load_checkpoint,
    load_local_qwen,
    load_salmonn_data,
    strip_markdown_fence,
    write_compact_salmonn_json,
)


DEFAULT_OUTPUT = "salmonn_data_v1.1/hq_music/youtube_crawled_gemini_music_captioning_segmented_40s_qwen3.5_35b_a3b_template_qa.json"
DEFAULT_NUM_QA = 3


def current_timestamp():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S%z")


def get_rank_id():
    for key in ("RANK_ID", "RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None and value != "":
            return value
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible_devices:
        return f"gpu:{cuda_visible_devices}"
    return "main"


def log_message(message, level="INFO"):
    print(f"[{current_timestamp()}][rank {get_rank_id()}][{level}] {message}", flush=True)


META_QA_TEMPLATES = [
    {
        "id": "holistic_style_mood",
        "name": "Holistic style / mood inference",
        "instruction": "Generate one QA pair that asks what overall style, mood, or expressive character can be inferred from the audio, using multiple musical cues as evidence.",
        "patterns": [
            r"\b(style|genre|mood|emotion|atmosphere|character|evokes|feeling|expressive)\b",
        ],
    },
    {
        "id": "contrastive_classification",
        "name": "Contrastive classification",
        "instruction": "Generate one QA pair that asks why the music is better described as one style, mood, or category rather than another plausible but incorrect one. The incorrect alternative must be ruled out using evidence explicitly present in the description.",
        "patterns": [
            r"\b(style|genre|mood|emotion|atmosphere|character)\b",
            r"\b(classical|jazz|rock|pop|folk|electronic|cinematic|soundtrack|ambient|dance|ballad|orchestral|instrumental)\b",
        ],
    },
    {
        "id": "temporal_structural_change",
        "name": "Temporal / structural change",
        "instruction": "Generate one QA pair about changes over time, such as shifts in energy, texture, intensity, groove, dynamics, or musical section. Only ask about temporal or structural change that is explicitly stated in the description.",
        "patterns": [
            r"\b(begins?|starts?|opens?|then|later|middle section|second half|first half|builds?|building|develops?|transitions?|shifts?|changes?|crescendo|decrescendo|climax|returns?|concludes?|ending|section|arc|journey|narrative|closure)\b",
        ],
    },
    {
        "id": "instrument_timbre_expression",
        "name": "Instrument / timbre / expression",
        "instruction": "Generate one QA pair that asks how a specific instrument's timbre, articulation, or playing style contributes to the emotional or stylistic effect.",
        "patterns": [
            r"\b(piano|guitar|drum|bass|violin|viola|cello|flute|saxophone|trumpet|trombone|clarinet|strings?|brass|woodwinds?|synthesizer|synth|vocal|voice|percussion|instrument)\b",
            r"\b(timbre|tone|articulation|playing style|breathy|bright|warm|resonant|mellow|crisp|distorted|legato|staccato|vibrato|plucked|bowed)\b",
        ],
    },
    {
        "id": "rhythm_dynamics_energy",
        "name": "Rhythm / dynamics / energy",
        "instruction": "Generate one QA pair focused on rhythm, tempo, dynamics, intensity, groove, or forward momentum, and explain how these features affect the music's energy.",
        "patterns": [
            r"\b(rhythm|tempo|beat|pulse|groove|meter|pace|dynamic|dynamics|crescendo|decrescendo|volume|intensity|energy|energetic|momentum|driving|steady|fast|slow|moderate|propulsive)\b",
        ],
    },
    {
        "id": "negative_evidence_rule_out",
        "name": "Negative evidence / rule-out",
        "instruction": "Generate one QA pair that asks what is absent from the audio and how that absence helps rule out another interpretation. Only use absences explicitly stated in the description.",
        "patterns": [
            r"\b(no |without|lacks?|absent|absence|does not|do not|not feature|purely instrumental|no lyrics|no vocals|exclusively|solo)\b",
        ],
    },
    {
        "id": "scene_functional_suitability",
        "name": "Scene / functional suitability",
        "instruction": "Generate one QA pair that asks what kind of scene, setting, or listening context the music would support, grounded strictly in musical evidence. Do not mention specific media, locations, or story events unless supported by the description.",
        "patterns": [
            r"\b(suitable|well-suited|background|scene|setting|context|film|cinematic|soundtrack|game|dance|ceremony|meditation|relaxation|accompanying|support)\b",
        ],
    },
    {
        "id": "performance_phrasing_nuance",
        "name": "Performance / phrasing / expressive nuance",
        "instruction": "Generate one QA pair about performance qualities such as articulation, phrasing, dynamic shaping, precision, expressiveness, or restraint.",
        "patterns": [
            r"\b(articulation|phrasing|phrase|expressive|expression|precision|restraint|subtle|delicate|nuance|rubato|legato|staccato|vibrato|crescendos?|decrescendos?|shaping|controlled|smooth|cleanly)\b",
        ],
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

Requirements for the question:
1. The question must begin with "<audio>".
2. The question should sound as if it is being asked about the audio itself, not about a caption or text.
3. The question should be analytical and require synthesizing multiple details from the description.
4. The question must stay within the specialized focus above.
5. Do not ask about unsupported facts such as exact key, composer, artist, recording venue, microphone placement, specific chord names, or external cultural references unless explicitly stated.

Requirements for the answer:
1. The answer must be fully supported by the provided description.
2. The answer should cite multiple musical cues from the description.
3. The answer should be concise but substantive, usually 2-5 sentences.
4. Do not introduce new facts beyond the description.
5. Do not mention that the answer is based on a caption or text description. Write as if the evidence comes from the audio.
6. Use cautious, evidence-grounded language. Avoid overclaiming beyond the provided description. In particular, do not use overly definitive words such as "clearly", "obviously", "definitively", "undoubtedly", or "explicitly proves" unless the description directly warrants that level of certainty. Prefer formulations such as "the audio suggests", "the musical evidence indicates", "this supports the interpretation that", or "the piece is better characterized as". The answer should sound confident but should not imply stronger certainty than the evidence allows.

Output format:
Return only valid JSON as one object:

{{
  "question": "<audio>...",
  "answer": "..."
}}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate template-guided music QA pairs from SALMONN-style music-caption data using Qwen."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SALMONN-style music caption JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output SALMONN-style QA JSON.")
    parser.add_argument("--tmp-output", default="", help="Checkpoint JSONL path. Defaults to <output>.tmp.jsonl.")
    parser.add_argument("--failed-output", default="", help="Failed-sample JSONL path. Defaults to <output>.failed.jsonl.")
    parser.add_argument("--backend", choices=("local", "openai"), default="local", help="Run local transformers inference or use an OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="", help="API key for OpenAI-compatible backend.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name exposed by the endpoint.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path or Hugging Face model id.")
    parser.add_argument("--num-qa", type=int, default=DEFAULT_NUM_QA, help="Maximum number of applicable templates sampled per item.")
    parser.add_argument("--seed", type=int, default=20260617, help="Seed for deterministic template sampling.")
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
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="Number of source samples to accumulate before one batched local generation call.",
    )
    parser.add_argument("--debug", action="store_true", help="Only process the first 5 input samples.")
    parser.add_argument("--limit", type=int, default=0, help="Optional positive sample limit. Overrides --debug if set.")
    parser.add_argument("--finalize-only", action="store_true", help="Only convert the checkpoint JSONL to final SALMONN JSON.")
    return parser.parse_args()


def template_is_applicable(template, caption):
    caption = caption.lower()
    return all(re.search(pattern, caption) for pattern in template["patterns"])


def get_applicable_templates(caption):
    return [
        template
        for template in META_QA_TEMPLATES
        if template_is_applicable(template, caption)
    ]


def sample_templates(caption, source_index, num_templates, seed):
    valid_templates = get_applicable_templates(caption)
    if not valid_templates:
        valid_templates = [META_QA_TEMPLATES[0]]
    if len(valid_templates) <= num_templates:
        return valid_templates
    rng = random.Random(seed + source_index)
    return rng.sample(valid_templates, num_templates)


def build_template_prompt(music_description, template):
    return PROMPT_TEMPLATE.format(
        music_description=music_description.strip(),
        template_instruction=template["instruction"],
    )


def parse_single_qa_response(text):
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
    if not question.startswith("<audio>"):
        question = "<audio>" + question.lstrip()
    return {"question": question, "answer": answer}


def finalize_output(tmp_path, output_path, source_count):
    records = load_checkpoint(tmp_path)
    missing = [idx for idx in range(source_count) if idx not in records]
    output_items = []
    for idx in sorted(records):
        output_items.extend(records[idx]["qa_items"])
    write_compact_salmonn_json(output_path, output_items)
    return len(output_items), len(missing)


def load_generator(args):
    if args.backend == "openai":
        if not args.base_url:
            raise ValueError("Missing --base-url for OpenAI-compatible backend.")
        if not args.api_key:
            raise ValueError("Missing --api-key for OpenAI-compatible backend.")
        from openai import OpenAI
        return OpenAI(base_url=args.base_url, api_key=args.api_key)
    return load_local_qwen(args)


def generate_one(generator, args, prompt):
    if args.backend == "openai":
        return call_qwen(generator, args, prompt)
    model, tokenizer = generator
    return call_local_qwen(model, tokenizer, args, prompt)


def generate_many(generator, args, prompts):
    if args.backend == "openai":
        return [call_qwen(generator, args, prompt) for prompt in prompts]
    model, tokenizer = generator
    return call_local_qwen_batch(model, tokenizer, args, prompts)


def generate_template_batch(generator, args, prompts, templates, source_index):
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_texts = generate_many(generator, args, prompts)
            if len(raw_texts) != len(prompts):
                raise ValueError(f"Expected {len(prompts)} batch outputs, got {len(raw_texts)}.")
            qas = [parse_single_qa_response(raw_text) for raw_text in raw_texts]
            return qas, []
        except Exception as exc:
            last_error = exc
            log_message(
                f"[WARN] source={source_index} batched generation "
                f"attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
                level="WARN",
            )
            print(traceback.format_exc(), flush=True)
            if attempt < args.max_retries:
                time.sleep(args.retry_sleep * attempt)

    template_errors = [
        {
            "template_id": template["id"],
            "error": repr(last_error),
            "batch_failed": True,
        }
        for template in templates
    ]
    return None, template_errors


def build_source_job(source_item, source_index, args):
    caption = get_caption(source_item, source_index)
    selected_templates = sample_templates(caption, source_index, args.num_qa, args.seed)
    prompts = [build_template_prompt(caption, template) for template in selected_templates]
    return {
        "source_index": source_index,
        "source_item": source_item,
        "selected_templates": selected_templates,
        "prompts": prompts,
    }


def generate_source_batch(generator, args, jobs):
    prompt_entries = []
    for job in jobs:
        for template, prompt in zip(job["selected_templates"], job["prompts"]):
            prompt_entries.append({
                "source_index": job["source_index"],
                "template": template,
                "prompt": prompt,
            })

    if not prompt_entries:
        return {}, []

    source_ids = ",".join(str(job["source_index"]) for job in jobs)
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            raw_texts = generate_many(
                generator,
                args,
                [entry["prompt"] for entry in prompt_entries],
            )
            if len(raw_texts) != len(prompt_entries):
                raise ValueError(f"Expected {len(prompt_entries)} batch outputs, got {len(raw_texts)}.")

            grouped_qas = {job["source_index"]: [] for job in jobs}
            for entry, raw_text in zip(prompt_entries, raw_texts):
                qa = parse_single_qa_response(raw_text)
                grouped_qas[entry["source_index"]].append((entry["template"], qa))
            return grouped_qas, []
        except Exception as exc:
            last_error = exc
            log_message(
                f"source_batch={source_ids} prompts={len(prompt_entries)} "
                f"attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
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


def process_one_source(generator, args, job):
    source_index = job["source_index"]
    source_item = job["source_item"]
    selected_templates = job["selected_templates"]
    prompts = job["prompts"]
    qa_items = []
    succeeded_template_ids = []
    template_errors = []

    qas, batch_errors = generate_template_batch(
        generator,
        args,
        prompts,
        selected_templates,
        source_index,
    )

    if qas is not None:
        for template, qa in zip(selected_templates, qas):
            qa_items.append(build_qa_salmonn_item(source_item, qa))
            succeeded_template_ids.append(template["id"])
        return qa_items, succeeded_template_ids, template_errors

    template_errors.extend(batch_errors)
    log_message(
        f"source={source_index} falling back to per-template generation",
        level="WARN",
    )
    for template, prompt in zip(selected_templates, prompts):
        last_error = None
        for attempt in range(1, args.max_retries + 1):
            try:
                raw_text = generate_one(generator, args, prompt)
                qa = parse_single_qa_response(raw_text)
                item = build_qa_salmonn_item(source_item, qa)
                qa_items.append(item)
                succeeded_template_ids.append(template["id"])
                break
            except Exception as exc:
                last_error = exc
                log_message(
                    f"source={source_index} template={template['id']} "
                    f"attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
                    level="WARN",
                )
                print(traceback.format_exc(), flush=True)
                if attempt < args.max_retries:
                    time.sleep(args.retry_sleep * attempt)
        else:
            template_errors.append({
                "template_id": template["id"],
                "error": repr(last_error),
            })
    return qa_items, succeeded_template_ids, template_errors


def write_job_result(tmp_f, failed_path, job, qa_items, succeeded_template_ids, template_errors, total_count):
    source_index = job["source_index"]
    selected_templates = job["selected_templates"]
    if qa_items:
        record = {
            "source_index": source_index,
            "template_ids": succeeded_template_ids,
            "qa_items": qa_items,
            "template_errors": template_errors,
        }
        tmp_f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        tmp_f.flush()
        log_message(
            f"source={source_index} progress={source_index + 1}/{total_count} "
            f"generated={len(qa_items)} selected_templates={len(selected_templates)} "
            f"template_ids={','.join(succeeded_template_ids)}"
        )
    else:
        append_failed_record(
            failed_path,
            {
                "source_index": source_index,
                "selected_template_ids": [template["id"] for template in selected_templates],
                "template_errors": template_errors,
            },
        )
        log_message(
            f"Skipping source index {source_index}; all selected templates failed. "
            f"Logged to {failed_path}",
            level="ERROR",
        )


def process_source_jobs(generator, args, jobs, tmp_f, failed_path, total_count):
    grouped_qas, batch_errors = generate_source_batch(generator, args, jobs)
    batch_errors_by_source = {}
    for error in batch_errors:
        batch_errors_by_source.setdefault(error["source_index"], []).append(error)

    if grouped_qas is None:
        log_message(
            f"source_batch={','.join(str(job['source_index']) for job in jobs)} "
            "falling back to per-source generation",
            level="WARN",
        )
        for job in jobs:
            qa_items, succeeded_template_ids, template_errors = process_one_source(generator, args, job)
            template_errors = batch_errors_by_source.get(job["source_index"], []) + template_errors
            write_job_result(
                tmp_f,
                failed_path,
                job,
                qa_items,
                succeeded_template_ids,
                template_errors,
                total_count,
            )
        return

    for job in jobs:
        source_item = job["source_item"]
        template_qa_pairs = grouped_qas[job["source_index"]]
        qa_items = [
            build_qa_salmonn_item(source_item, qa)
            for _, qa in template_qa_pairs
        ]
        succeeded_template_ids = [
            template["id"]
            for template, _ in template_qa_pairs
        ]
        write_job_result(
            tmp_f,
            failed_path,
            job,
            qa_items,
            succeeded_template_ids,
            [],
            total_count,
        )


def main():
    args = parse_args()
    if args.sample_batch_size < 1:
        raise ValueError("--sample-batch-size must be at least 1.")

    input_path = Path(args.input)
    output_path = Path(args.output)
    tmp_path = Path(args.tmp_output) if args.tmp_output else Path(str(output_path) + ".tmp.jsonl")
    failed_path = Path(args.failed_output) if args.failed_output else Path(str(output_path) + ".failed.jsonl")

    data = load_salmonn_data(input_path)
    limit = args.limit if args.limit > 0 else (5 if args.debug else len(data))
    limit = min(limit, len(data))
    data = data[:limit]

    if args.finalize_only:
        total, missing = finalize_output(tmp_path, output_path, len(data))
        log_message(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")
        return

    generator = load_generator(args)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(tmp_path)

    with tmp_path.open("a", encoding="utf-8") as tmp_f:
        pending_jobs = []
        for source_index, source_item in enumerate(data):
            if source_index in checkpoint:
                continue

            pending_jobs.append(build_source_job(source_item, source_index, args))
            if len(pending_jobs) >= args.sample_batch_size:
                process_source_jobs(generator, args, pending_jobs, tmp_f, failed_path, len(data))
                pending_jobs = []

        if pending_jobs:
            process_source_jobs(generator, args, pending_jobs, tmp_f, failed_path, len(data))

    total, missing = finalize_output(tmp_path, output_path, len(data))
    log_message(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")


if __name__ == "__main__":
    main()
