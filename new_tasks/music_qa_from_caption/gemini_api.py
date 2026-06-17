import argparse
import json
import os
import time
import traceback
from pathlib import Path

from qwen_music_qa_pipeline import (
    DEFAULT_INPUT,
    DEFAULT_NUM_QA,
    build_prompt,
    build_qa_salmonn_item,
    get_caption,
    append_failed_record,
    load_checkpoint,
    load_salmonn_data,
    parse_qa_response,
    write_compact_salmonn_json,
)


DEFAULT_OUTPUT = "salmonn_data_v1.1/youtube_crawled_gemini_music_captioning_segmented_40s_gemini_qa.json"
DEFAULT_MODEL = "gemini-2.5-pro-thinking-2048"
DEFAULT_BASE_URL = "http://35.220.164.252:3888/v1/"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SALMONN-style music QA data from a SALMONN music-caption manifest using a Gemini OpenAI-compatible endpoint."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input SALMONN-style music caption JSON.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output SALMONN-style QA JSON.")
    parser.add_argument("--tmp-output", default="", help="Checkpoint JSONL path. Defaults to <output>.tmp.jsonl.")
    parser.add_argument("--failed-output", default="", help="Failed-sample JSONL path. Defaults to <output>.failed.jsonl.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-qa", type=int, default=DEFAULT_NUM_QA)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--debug", action="store_true", help="Only process the first 5 input samples.")
    parser.add_argument("--limit", type=int, default=0, help="Optional positive sample limit. Overrides --debug if set.")
    parser.add_argument("--finalize-only", action="store_true", help="Only convert the checkpoint JSONL to final SALMONN JSON.")
    return parser.parse_args()


def call_gemini(client, args, prompt):
    response = client.chat.completions.create(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    return response.choices[0].message.content


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

    if not args.api_key:
        raise ValueError("Missing --api-key or OPENAI_API_KEY.")

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
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
                    raw_text = call_gemini(client, args, prompt)
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
