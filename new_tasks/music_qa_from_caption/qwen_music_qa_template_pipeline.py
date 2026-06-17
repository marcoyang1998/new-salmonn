import argparse
import json
import random
import re
import time
import traceback
from pathlib import Path

from qwen_music_qa_pipeline import (
    DEFAULT_INPUT,
    DEFAULT_MODEL,
    DEFAULT_MODEL_PATH,
    append_failed_record,
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


def main():
    args = parse_args()
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
        print(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")
        return

    generator = load_generator(args)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(tmp_path)

    with tmp_path.open("a", encoding="utf-8") as tmp_f:
        for source_index, source_item in enumerate(data):
            if source_index in checkpoint:
                continue

            caption = get_caption(source_item, source_index)
            selected_templates = sample_templates(caption, source_index, args.num_qa, args.seed)
            qa_items = []
            succeeded_template_ids = []
            template_errors = []

            for template in selected_templates:
                prompt = build_template_prompt(caption, template)
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
                        print(
                            f"[WARN] source={source_index} template={template['id']} "
                            f"attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
                            flush=True,
                        )
                        print(traceback.format_exc(), flush=True)
                        if attempt < args.max_retries:
                            time.sleep(args.retry_sleep * attempt)
                else:
                    template_errors.append({
                        "template_id": template["id"],
                        "error": repr(last_error),
                    })

            if qa_items:
                record = {
                    "source_index": source_index,
                    "template_ids": succeeded_template_ids,
                    "qa_items": qa_items,
                    "template_errors": template_errors,
                }
                tmp_f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                tmp_f.flush()
                print(
                    f"[{source_index + 1}/{len(data)}] generated {len(qa_items)} QA items "
                    f"from {len(selected_templates)} selected templates",
                    flush=True,
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
                print(
                    f"[ERROR] Skipping source index {source_index}; all selected templates failed. "
                    f"Logged to {failed_path}",
                    flush=True,
                )

    total, missing = finalize_output(tmp_path, output_path, len(data))
    print(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")


if __name__ == "__main__":
    main()
