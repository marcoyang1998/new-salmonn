import argparse
import json
import random
import re
import time
import traceback
from pathlib import Path


DEFAULT_INPUT = "salmonn_data_v1.1/changli_data/Audioset_Qwen3OmniCap_train_filter_cos_salmonn.json"
DEFAULT_OUTPUT = "salmonn_data_v1.1/stage2/audioset_qwen3.5_35b_a3b_template_qa.json"
DEFAULT_MODEL = "Qwen3.5-35B-A3B"
DEFAULT_MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
DEFAULT_NUM_QA = 3


META_QA_TEMPLATES = [
    {
        "id": "acoustic_environment_space",
        "name": "Acoustic environment / physical space inference",
        "instruction": "Generate one QA pair that asks what the acoustic evidence suggests about the physical environment or recording space, such as indoor versus outdoor setting, reverberation, openness, reflective surfaces, distance, or microphone proximity.",
        "required_any": [
            [r"\b(environment|setting|space|room|hall|venue|indoor|outdoor|open-air|open air|enclosed|domestic|public|street|rural|urban|background)\b",
             r"\b(reverb|reverberation|reverberant|echo|acoustic|reflective|distant|close|microphone|wind)\b"],
        ],
    },
    {
        "id": "temporal_event_progression",
        "name": "Temporal event progression",
        "instruction": "Generate one QA pair about the sequence of audible events over time. The question should require reasoning from how sounds begin, overlap, transition, intensify, fade, or end.",
        "required_any": [
            [r"\b(begins?|starts?|opens?|initially|then|followed by|afterward|afterwards|subsequently|next|as .* continues|toward the end|ends?|closes?|concludes?|fades?|cuts off|abrupt|sequence|progression|transition)\b"],
        ],
    },
    {
        "id": "sound_source_activity_inference",
        "name": "Sound source / activity inference",
        "instruction": "Generate one QA pair that asks what specific activity, event, or sound source can be inferred from multiple audible cues. The answer should connect the sound-source evidence to the inferred context.",
        "required_any": [
            [r"\b(sound|sounds|event|activity|source|object|machine|vehicle|engine|water|door|impact|applause|footsteps|animal|music|instrument|tool|device|mechanical|electronic)\b"],
            [r"\b(suggest|suggests|indicate|indicates|imply|implies|likely|probably|consistent with|characteristic of|context|scene)\b"],
        ],
    },
    {
        "id": "speech_semantics_context",
        "name": "Speech semantics + acoustic context",
        "instruction": "Generate one QA pair that combines spoken content, speaker delivery, and surrounding sounds to infer the situation, intent, relationship, or meaning of the speech.",
        "required_any": [
            [r"\b(speaker|voice|voices|speech|says|said|phrase|words?|utterance|conversation|commentary|narration|speaks?|asks?|exclaims?|language)\b"],
            [r"\b(tone|delivery|emotion|calm|excited|playful|formal|instructional|background|environment|context|surrounding|responds?|interaction)\b"],
        ],
    },
    {
        "id": "interaction_between_sources",
        "name": "Interaction between sound sources",
        "instruction": "Generate one QA pair that asks what interaction or relationship can be inferred between sound sources, such as people with animals, speakers with objects, vehicles with surfaces, or simultaneous foreground and background events.",
        "required_any": [
            [r"\b(interaction|interact|responds?|reaction|causing|causes|alongside|overlap|simultaneously|concurrently|foreground|background|with|among)\b"],
            [r"\b(people|person|speaker|child|adult|animal|dog|bird|goat|sheep|vehicle|machine|object|music|crowd|audience)\b"],
        ],
    },
    {
        "id": "emotion_paralinguistic_inference",
        "name": "Emotion / paralinguistic inference",
        "instruction": "Generate one QA pair about emotional state, attitude, urgency, humor, tension, surprise, calmness, or other paralinguistic meaning inferred from vocal delivery and non-speech acoustic cues.",
        "required_any": [
            [r"\b(emotion|emotional|tone|mood|attitude|urgency|urgent|calm|excited|playful|amused|surprised|fear|tense|suspense|joy|anger|frustration|awe|relaxed|enthusiastic|paralinguistic)\b"],
        ],
    },
    {
        "id": "negative_evidence_rule_out",
        "name": "Negative evidence / rule-out",
        "instruction": "Generate one QA pair that asks what is absent or not supported by the audio, and how that absence helps rule out another interpretation. Only use absences explicitly stated in the description.",
        "required_any": [
            [r"\b(no |without|lacks?|absent|absence|no evidence|not present|does not|do not|none|free from|void of|only|sole|single|entirely)\b"],
        ],
    },
    {
        "id": "technical_recording_artifacts",
        "name": "Technical / recording artifact reasoning",
        "instruction": "Generate one QA pair about recording quality or technical acoustic artifacts, such as hiss, clipping, distortion, wind buffeting, microphone handling, abrupt cutoff, reverb, fidelity, or production quality, and what they imply about the recording.",
        "required_any": [
            [r"\b(hiss|static|clipping|distortion|artifact|microphone|mic|buffeting|wind noise|recording quality|fidelity|compressed|mono|stereo|abruptly|truncated|cuts off|cut off|reverb|reverberation|clean|clear|consumer-grade|studio|production)\b"],
        ],
    },
]


PROMPT_TEMPLATE = """
Role: You are an expert in audio scene understanding and dataset construction for audio-language models.

Task: You will be given a detailed natural-language audio description and one specialized QA-generation focus. Based only on information explicitly contained in the description, generate exactly one high-quality audio-grounded question-answer pair in the style of analytical Gemini AudioSet QA.

Input:
A detailed audio description:
{audio_description}

Specialized QA focus:
{template_instruction}

Requirements for the question:
1. The question must begin with "<audio>".
2. The question should sound as if it is being asked about the audio itself, not about a caption or text.
3. The question should be analytical and require synthesizing multiple audible cues, not a simple sound-label lookup.
4. The question must stay within the specialized focus above.
5. Prefer multi-cue formulations such as "Considering...", "By analyzing...", "Given...", or "Based on..." when natural.
6. Do not ask about unsupported facts such as exact identities, exact locations, recording equipment models, speaker names, or external cultural details unless explicitly stated.

Requirements for the answer:
1. The answer must be fully supported by the provided description.
2. The answer should explain the reasoning by citing multiple audible cues from the description.
3. The answer should be concise but substantive, usually 2-5 sentences.
4. Do not introduce new facts beyond the description.
5. Do not mention that the answer is based on a caption or text description. Write as if the evidence comes from the audio.
6. Use cautious, evidence-grounded language. Avoid overclaiming beyond the provided description. In particular, do not use overly definitive words such as "clearly", "obviously", "definitively", "undoubtedly", or "explicitly proves" unless the description directly warrants that level of certainty. Prefer formulations such as "the audio suggests", "the acoustic evidence indicates", "this supports the interpretation that", or "the recording is better characterized as". The answer should sound confident but should not imply stronger certainty than the evidence allows.

Output format:
Return only valid JSON as one object:

{{
  "question": "<audio>...",
  "answer": "..."
}}
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate template-guided AudioSet QA pairs from detailed SALMONN-style captions using Qwen."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SALMONN-style detailed AudioSet caption JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output SALMONN-style QA JSON.")
    parser.add_argument("--tmp-output", default="", help="Checkpoint JSONL path. Defaults to <output>.tmp.jsonl.")
    parser.add_argument("--failed-output", default="", help="Failed-sample JSONL path. Defaults to <output>.failed.jsonl.")
    parser.add_argument("--backend", choices=("local", "openai"), default="local", help="Run local transformers inference or use an OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="", help="API key for OpenAI-compatible backend.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name exposed by the endpoint.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path or Hugging Face model id.")
    parser.add_argument("--num-qa", type=int, default=3, help="Maximum number of applicable templates sampled per item.")
    parser.add_argument("--seed", type=int, default=20260618, help="Seed for deterministic template sampling.")
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
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of logical data shards.")
    parser.add_argument("--shard-id", type=int, default=0, help="This worker's logical shard id in [0, num_shards).")
    parser.add_argument("--debug", action="store_true", help="Only process the first 5 input samples.")
    parser.add_argument("--limit", type=int, default=0, help="Optional positive sample limit. Overrides --debug if set.")
    parser.add_argument("--finalize-only", action="store_true", help="Only convert the checkpoint JSONL to final SALMONN JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Only report applicable-template statistics; do not load or call Qwen.")
    return parser.parse_args()


def stream_salmonn_items(path, limit=0):
    decoder = json.JSONDecoder()
    chunk_size = 1024 * 1024
    buffer = ""
    found_array = False
    count = 0

    with open(path, "r", encoding="utf-8") as f:
        while not found_array:
            chunk = f.read(chunk_size)
            if not chunk:
                raise ValueError(f"Could not find top-level data array in {path}")
            buffer += chunk
            start = buffer.find("[")
            if start != -1:
                buffer = buffer[start + 1 :]
                found_array = True

        while True:
            buffer = buffer.lstrip()
            if buffer.startswith("]"):
                return
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue

            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    yield count, item
                    count += 1
                    if limit and count >= limit:
                        return
                    buffer = buffer[end:]
                    break
                except json.JSONDecodeError:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        raise
                    buffer += chunk


def iter_shard_items(path, num_shards=1, shard_id=0, limit=0):
    yielded = 0
    for source_index, item in stream_salmonn_items(path):
        if source_index % num_shards != shard_id:
            continue
        yield source_index, item
        yielded += 1
        if limit and yielded >= limit:
            return


def count_salmonn_items(path, limit=0):
    count = 0
    for count, _ in stream_salmonn_items(path, limit=limit):
        pass
    return count + 1 if count or limit else 0


def get_caption(item, index):
    messages = item.get("messages", [])
    if len(messages) != 2:
        raise ValueError(f"Input item {index} must have exactly 2 messages.")
    caption = messages[1].get("content", "").strip()
    if not caption:
        raise ValueError(f"Input item {index} has an empty assistant caption.")
    return caption


def template_is_applicable(template, caption):
    caption = caption.lower()
    return all(
        any(re.search(pattern, caption) for pattern in pattern_group)
        for pattern_group in template["required_any"]
    )


def get_applicable_templates(caption):
    return [
        template
        for template in META_QA_TEMPLATES
        if template_is_applicable(template, caption)
    ]


def sample_templates(caption, source_index, num_templates, seed):
    valid_templates = get_applicable_templates(caption)
    if not valid_templates:
        valid_templates = [META_QA_TEMPLATES[2]]
    if len(valid_templates) <= num_templates:
        return valid_templates
    rng = random.Random(seed + source_index)
    return rng.sample(valid_templates, num_templates)


def build_template_prompt(audio_description, template):
    return PROMPT_TEMPLATE.format(
        audio_description=audio_description.strip(),
        template_instruction=template["instruction"],
    )


def strip_markdown_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


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


def build_qa_salmonn_item(source_item, qa):
    item = {
        "messages": [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ],
        "audios": source_item["audios"],
        "durations": source_item["durations"],
        "task_type": "QA",
    }
    for key in ("start_time", "end_time"):
        if key in source_item:
            item[key] = source_item[key]
    return item


def load_checkpoint_indices(path):
    indices = set()
    if not path.exists():
        return indices
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            indices.add(int(record["source_index"]))
    return indices


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def load_local_qwen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "auto": "auto",
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    model_kwargs = {
        "torch_dtype": dtype_map[args.torch_dtype],
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.eval()
    return model, tokenizer


def call_local_qwen(model, tokenizer, args, prompt):
    import torch

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        generation_kwargs.update({
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
        })
        if args.min_p > 0:
            generation_kwargs["min_p"] = args.min_p

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    generated_ids = output_ids[:, inputs.input_ids.shape[-1]:]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def call_qwen(client, args, prompt):
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content


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


def write_final_json_from_checkpoint(tmp_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_items = 0
    seen = set()
    with output_path.open("w", encoding="utf-8") as out_f:
        out_f.write('{"data": [\n')
        first = True
        if tmp_path.exists():
            with tmp_path.open("r", encoding="utf-8") as tmp_f:
                for line in tmp_f:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    source_index = int(record["source_index"])
                    if source_index in seen:
                        continue
                    seen.add(source_index)
                    for item in record.get("qa_items", []):
                        if not first:
                            out_f.write(",\n")
                        out_f.write(json.dumps(item, ensure_ascii=False, separators=(", ", ": ")))
                        first = False
                        total_items += 1
        out_f.write("\n]}\n")
    return total_items, len(seen)


def dry_run(args):
    from collections import Counter

    valid_count_dist = Counter()
    template_counts = Counter()
    selected_counts = Counter()
    total = 0
    total_valid = 0
    limit = args.limit if args.limit > 0 else (5 if args.debug else 0)
    for source_index, item in iter_shard_items(args.input, args.num_shards, args.shard_id, limit=limit):
        caption = get_caption(item, source_index)
        valid = get_applicable_templates(caption)
        selected = sample_templates(caption, source_index, args.num_qa, args.seed)
        valid_count_dist[len(valid)] += 1
        total_valid += len(valid)
        total += 1
        for template in valid:
            template_counts[template["id"]] += 1
        for template in selected:
            selected_counts[template["id"]] += 1

    print(f"samples: {total}")
    print(f"avg_valid_templates: {total_valid / total if total else 0:.4f}")
    print("valid_count_distribution:")
    for count, n in sorted(valid_count_dist.items()):
        print(f"  {count}: {n} ({n / total * 100:.2f}%)")
    print("template_applicability:")
    for template in META_QA_TEMPLATES:
        n = template_counts[template["id"]]
        print(f"  {template['id']}: {n} ({n / total * 100:.2f}%)")
    print("sampled_template_counts:")
    for template_id, n in selected_counts.most_common():
        print(f"  {template_id}: {n}")


def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("--shard-id must satisfy 0 <= shard_id < num_shards")

    input_path = Path(args.input)
    output_path = Path(args.output)
    tmp_path = Path(args.tmp_output) if args.tmp_output else Path(str(output_path) + ".tmp.jsonl")
    failed_path = Path(args.failed_output) if args.failed_output else Path(str(output_path) + ".failed.jsonl")

    if args.dry_run:
        dry_run(args)
        return

    limit = args.limit if args.limit > 0 else (5 if args.debug else 0)

    if args.finalize_only:
        total_items, completed_sources = write_final_json_from_checkpoint(tmp_path, output_path)
        print(f"Wrote {total_items} QA training items from {completed_sources} source items to {output_path}")
        return

    generator = load_generator(args)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_indices = load_checkpoint_indices(tmp_path)

    processed_this_run = 0
    for source_index, source_item in iter_shard_items(input_path, args.num_shards, args.shard_id, limit=limit):
        if source_index in checkpoint_indices:
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
                    qa_items.append(build_qa_salmonn_item(source_item, qa))
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
            append_jsonl(
                tmp_path,
                {
                    "source_index": source_index,
                    "template_ids": succeeded_template_ids,
                    "qa_items": qa_items,
                    "template_errors": template_errors,
                },
            )
            checkpoint_indices.add(source_index)
            processed_this_run += 1
            print(
                f"[{source_index}] generated {len(qa_items)} QA items "
                f"from {len(selected_templates)} selected templates",
                flush=True,
            )
        else:
            append_jsonl(
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

    total_items, completed_sources = write_final_json_from_checkpoint(tmp_path, output_path)
    print(
        f"Wrote {total_items} QA training items from {completed_sources} source items to {output_path}; "
        f"processed this run: {processed_this_run}",
        flush=True,
    )


if __name__ == "__main__":
    main()
