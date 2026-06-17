import argparse
import json
import os
import re
import time
import traceback
from pathlib import Path

DEFAULT_INPUT = "salmonn_data_v1.1/hq_music/youtube_crawled_gemini_music_captioning_segmented_40s.json"
DEFAULT_OUTPUT = "salmonn_data_v1.1/hq_music/youtube_crawled_gemini_music_captioning_segmented_40s_qwen3.5_35b_a3b_qa.json"
# DEFAULT_MODEL = "Qwen3.5-35B-A3B-FP8"
DEFAULT_MODEL = "Qwen3.5-35B-A3B"
# DEFAULT_MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/0b2752837483aa34b3db6e83e151b150c0e00e49"
DEFAULT_MODEL_PATH = "/mnt/shared-storage-gpfs2/gpfs2-shared-public/huggingface/hub/models--Qwen--Qwen3.5-35B-A3B/snapshots/ec2d4ece1ffb563322cbee9a48fe0e3fcbce0307"
DEFAULT_NUM_QA = 3

PROMPT_TEMPLATE = """
Role: You are a world-class expert in music analysis, audio understanding, and dataset construction for audio-language models.

Task: You will be given a detailed natural-language music description. Based only on the information explicitly contained in this description, generate high-quality audio/music-grounded question-answer pairs. These QA pairs should be suitable for training a speech/audio language model.

Input:
A detailed music description:
{music_description}

Your goal:
Generate exactly {num_qa} high-quality QA pairs that require the model to synthesize multiple musical details from the description. The questions should sound as if they are being asked about the audio itself, and every question must begin with "<audio>". The answers must be strictly grounded in the provided music description.

Requirements for the questions:

1. The question should be challenging and analytical, not a simple attribute lookup.
2. The question should require combining at least two aspects of the music, such as instrumentation, timbre, melody, harmony, rhythm, tempo, dynamics, style, mood, texture, or expressive character.
3. The question should be objective and answerable from the provided description alone.
4. Avoid simple questions such as "What instrument is playing?" or "What is the mood?"
5. Avoid asking about information not supported by the description, such as exact key, composer, artist, recording venue, microphone placement, specific chord names, or external cultural references, unless they are explicitly stated.
6. The question should be phrased naturally as an audio/music understanding question.

Requirements for the answers:

1. The answer must be fully supported by the provided description.
2. The answer should explain the reasoning by citing multiple musical cues from the description.
3. The answer should be concise but substantive, usually 2-5 sentences.
4. Do not introduce new facts beyond the description.
5. Do not mention that the answer is based on a caption or text description. Write as if the evidence comes from the audio.
6. Avoid adding general musicological claims unless they are directly supported by the caption. Prefer rephrasing and synthesizing the given evidence over introducing new explanatory details.

When generating multiple QA pairs, choose distinct reasoning directions from the following list. Do not use the same reasoning direction more than once for the same caption.

Possible reasoning directions:
1. Holistic inference: infer the overall style, mood, or expressive character from multiple musical cues.
2. Contrastive reasoning: explain why the music fits one style, mood, or setting better than another plausible alternative.
3. Structural evidence: analyze how rhythm, harmony, melody, dynamics, timbre, texture, or instrumentation support a specific interpretation.
4. Negative evidence: identify what is absent from the audio and how that absence helps rule out other interpretations, such as vocals, drums, dense ensemble texture, or high-energy production.
5. Functional / scene inference: infer what kind of scene, background use, or listening context the music would support, grounded only in the musical evidence.
6. Temporal / phrase-shaping analysis: explain how tempo, rhythmic consistency, crescendos, decrescendos, or phrase shaping affect the listener's perception over time.
7. Performer / acoustic presentation analysis: discuss how articulation, timbre, dynamics, and texture affect the perceived intimacy, clarity, or expressive quality of the performance.
8. Ranking or emphasis question: ask which musical cue is most important for a particular effect, and explain why.

The three questions must not share the same surface pattern. In particular, do not make more than one question use the form "How do X and Y work together/contribute to Z?"

Question surface-form diversity:
Do not overuse "How do/does..." questions. For each set of 3 QA pairs, at most one question may begin with "How do", "How does", "How can", or "In what ways".

Use varied question forms. Prefer questions beginning with different patterns, such as:
- What evidence suggests that ...
- Why is this piece better described as ... rather than ...?
- What makes the music feel ...?
- Which musical cues indicate ...?
- What can be inferred about ... from ...?
- Why would this music be suitable for ...?
- What aspects of the audio rule out ...?
- Which feature is most responsible for ..., and why?
- How does ...?

For exactly 3 QA pairs, use three different surface forms. At least one question should involve contrastive or negative-evidence reasoning. At most one question should ask how multiple features work together.

Do not force every caption to use all of these directions. Choose the most suitable and non-overlapping directions based on the information available in the provided description.

Output format:
Return only valid JSON as a list of objects:

[
  {{
    "question": "<audio>...",
    "answer": "..."
  }},
  {{
    "question": "<audio>...",
    "answer": "..."
  }}
]
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate music QA pairs from SALMONN-style music-caption data using a Qwen OpenAI-compatible endpoint."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SALMONN-style music caption JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output SALMONN-style QA JSON.")
    parser.add_argument("--tmp-output", default="", help="Checkpoint JSONL path. Defaults to <output>.tmp.jsonl.")
    parser.add_argument("--failed-output", default="", help="Failed-sample JSONL path. Defaults to <output>.failed.jsonl.")
    parser.add_argument("--backend", choices=("local", "openai"), default="local", help="Run local transformers inference or use an OpenAI-compatible endpoint.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""), help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""), help="API key, preferably via OPENAI_API_KEY.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Qwen model name exposed by the endpoint.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Local Qwen model path or Hugging Face model id.")
    parser.add_argument("--num-qa", type=int, default=DEFAULT_NUM_QA, help="Number of QA pairs per input item.")
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


def load_salmonn_data(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj["data"]
    if not isinstance(data, list):
        raise ValueError(f"Expected top-level data list in {path}")
    return data


def get_caption(item, index):
    messages = item.get("messages", [])
    if len(messages) != 2:
        raise ValueError(f"Input item {index} must have exactly 2 messages.")
    caption = messages[1].get("content", "").strip()
    if not caption:
        raise ValueError(f"Input item {index} has an empty assistant caption.")
    return caption


def build_prompt(music_description, num_qa):
    return PROMPT_TEMPLATE.format(music_description=music_description.strip(), num_qa=num_qa)


def strip_markdown_fence(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_qa_response(text, expected_num_qa):
    text = strip_markdown_fence(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])

    if isinstance(parsed, dict):
        for key in ("data", "qas", "qa_pairs", "questions"):
            if key in parsed:
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got {type(parsed).__name__}.")
    if len(parsed) != expected_num_qa:
        raise ValueError(f"Expected {expected_num_qa} QA pairs, got {len(parsed)}.")

    qa_pairs = []
    for idx, qa in enumerate(parsed):
        if not isinstance(qa, dict):
            raise ValueError(f"QA item {idx} is not an object.")
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        if not question or not answer:
            raise ValueError(f"QA item {idx} has an empty question or answer.")
        if not question.startswith("<audio>"):
            question = "<audio>" + question.lstrip()
        qa_pairs.append({"question": question, "answer": answer})
    return qa_pairs


def call_qwen(client, args, prompt):
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content


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
    from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList

    class GeneratedPresencePenaltyLogitsProcessor(LogitsProcessor):
        def __init__(self, penalty, prompt_length):
            self.penalty = float(penalty)
            self.prompt_length = int(prompt_length)

        def __call__(self, input_ids, scores):
            if self.penalty == 0 or input_ids.shape[-1] <= self.prompt_length:
                return scores
            generated_ids = input_ids[:, self.prompt_length :]
            for batch_idx, token_ids in enumerate(generated_ids):
                scores[batch_idx, token_ids.unique()] -= self.penalty
            return scores

    messages = [{"role": "user", "content": prompt}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            enable_thinking=False,
        )
    except TypeError:
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    device = model.device if hasattr(model, "device") else next(model.parameters()).device
    inputs = inputs.to(device)
    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"]
        generate_inputs = inputs
    else:
        input_ids = inputs
        generate_inputs = {"input_ids": inputs}
    prompt_length = input_ids.shape[-1]

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
    if args.presence_penalty != 0:
        generation_kwargs["logits_processor"] = LogitsProcessorList([
            GeneratedPresencePenaltyLogitsProcessor(args.presence_penalty, prompt_length)
        ])

    with torch.inference_mode():
        output_ids = model.generate(**generate_inputs, **generation_kwargs)
    generated_ids = output_ids[0, prompt_length:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def build_qa_salmonn_item(source_item, qa):
    item = {
        "messages": [
            {"role": "user", "content": qa["question"]},
            {"role": "assistant", "content": qa["answer"]},
        ],
        "audios": source_item["audios"],
        "durations": source_item["durations"],
    }
    for key in ("start_time", "end_time"):
        if key in source_item:
            item[key] = source_item[key]
    return item


def load_checkpoint(path):
    records = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            records[int(record["source_index"])] = record
    return records


def append_failed_record(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_compact_salmonn_json(path, items):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write('{"data": [\n')
        for i, item in enumerate(items):
            f.write(json.dumps(item, ensure_ascii=False, separators=(", ", ": ")))
            f.write(",\n" if i < len(items) - 1 else "\n")
        f.write("]}\n")


def finalize_output(tmp_path, output_path, source_count, expected_num_qa, require_complete=False):
    records = load_checkpoint(tmp_path)
    missing = [idx for idx in range(source_count) if idx not in records]
    if missing and require_complete:
        raise RuntimeError(
            f"Checkpoint is incomplete: {len(missing)} missing source items. "
            f"First missing index: {missing[0]}"
        )

    output_items = []
    for idx in sorted(records):
        qa_items = records[idx]["qa_items"]
        if len(qa_items) != expected_num_qa:
            raise ValueError(f"Checkpoint item {idx} has {len(qa_items)} QA items.")
        output_items.extend(qa_items)
    write_compact_salmonn_json(output_path, output_items)
    return len(output_items), len(missing)


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
        total, missing = finalize_output(tmp_path, output_path, len(data), args.num_qa)
        print(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")
        return

    if args.backend == "openai":
        if not args.base_url:
            raise ValueError("Missing --base-url or OPENAI_BASE_URL.")
        if not args.api_key:
            raise ValueError("Missing --api-key or OPENAI_API_KEY.")
        from openai import OpenAI
        generator = OpenAI(base_url=args.base_url, api_key=args.api_key)
    else:
        generator = load_local_qwen(args)

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = load_checkpoint(tmp_path)

    with tmp_path.open("a", encoding="utf-8") as tmp_f:
        for index, source_item in enumerate(data):
            if index in checkpoint:
                continue

            prompt = build_prompt(get_caption(source_item, index), args.num_qa)
            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    if args.backend == "openai":
                        raw_text = call_qwen(generator, args, prompt)
                    else:
                        model, tokenizer = generator
                        raw_text = call_local_qwen(model, tokenizer, args, prompt)
                    qa_pairs = parse_qa_response(raw_text, args.num_qa)
                    qa_items = [build_qa_salmonn_item(source_item, qa) for qa in qa_pairs]
                    record = {
                        "source_index": index,
                        "qa_items": qa_items,
                    }
                    tmp_f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    tmp_f.flush()
                    print(f"[{index + 1}/{len(data)}] generated {len(qa_items)} QA items", flush=True)
                    break
                except Exception as exc:
                    last_error = exc
                    print(
                        f"[WARN] index={index} attempt={attempt}/{args.max_retries} failed: {repr(exc)}",
                        flush=True,
                    )
                    print(traceback.format_exc(), flush=True)
                    if attempt < args.max_retries:
                        time.sleep(args.retry_sleep * attempt)
            else:
                append_failed_record(
                    failed_path,
                    {
                        "source_index": index,
                        "error": repr(last_error),
                    },
                )
                print(
                    f"[ERROR] Skipping source index {index} after {args.max_retries} failed attempts. "
                    f"Logged to {failed_path}",
                    flush=True,
                )

    total, missing = finalize_output(tmp_path, output_path, len(data), args.num_qa)
    print(f"Wrote {total} QA training items to {output_path}; missing/skipped source items: {missing}")


if __name__ == "__main__":
    main()
