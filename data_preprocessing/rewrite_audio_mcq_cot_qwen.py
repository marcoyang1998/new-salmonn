#!/usr/bin/env python3
"""Rewrite verbose AudioMCQ Gemini CoT rationales with Qwen3-8B.

The script reads a SALMONN-style JSONL file whose assistant message contains a
verbose answer, asks Qwen3-8B to rewrite that answer into a concise
evidence-based rationale, and writes a JSONL file with the assistant content
replaced by the cleaned rationale.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT = Path("salmonn_data_v1.1/audio_mcq/audiomcq_strongac_gemini_cot.jsonl")
DEFAULT_OUTPUT = Path("salmonn_data_v1.1/audio_mcq/audiomcq_strongac_qwen_rationale.jsonl")
DEFAULT_MODEL = Path("/data/milsrg1/huggingface/cache/xy316/models--Qwen--Qwen3-8B")

SYSTEM_PROMPT = """You are an expert dataset editor.

Your task is to convert a verbose chain-of-thought style answer into a concise,
evidence-based rationale suitable for supervised fine-tuning.

Requirements:
1. Preserve the original conclusion and answer letter exactly.
2. Preserve all relevant observations and evidence from the original answer.
3. Preserve only the reasoning that links the evidence to the conclusion.
4. Remove conversational filler, self-talk, and meta-reasoning.
5. Do NOT introduce any new information that is not present in the original answer.
6. Do NOT speculate beyond the original reasoning.
7. Write in an objective explanatory style.
8. Preserve as much audio evidence as possible while removing filler and meta-reasoning.
9. The rewritten rationale should typically be 60-150 words.
10. Prefer retaining concrete observations about the audio rather than aggressively shortening the explanation.
11. End with "Final answer: <LETTER>. <option text>".

Remove wording such as:
- "Okay, so ..."
- "The task is straightforward."
- "I need to ..."
- "First, I ..."
- "Let's break/listen/analyze/assess ..."
- "The user wants me to ..."
- "My Analysis of the Audio ..."
- "Now let's assess the options."
- Markdown section headers, HTML/XML answer tags, and code fences.

The goal is to transform chain-of-thought into a compact evidence-based
explanation, not to perform new reasoning."""


ANSWER_PATTERNS = [
    re.compile(r"(?:so\s+)?the\s+answer\s+is\s+\**\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"final\s+answer\s*(?:is|:)\s+\**\(?([A-D])\)?", re.IGNORECASE),
    re.compile(r"(?:option|choice)\s+\**\(?([A-D])\)?\s+(?:is|fits|matches)", re.IGNORECASE),
]


def resolve_hf_model_path(path: str) -> str:
    """Resolve either a plain HF model directory, cache-layout model dir, or ID."""

    model_path = Path(path)
    if not model_path.exists():
        return path
    if (model_path / "config.json").exists():
        return str(model_path)

    refs_main = model_path / "refs" / "main"
    snapshots = model_path / "snapshots"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot = snapshots / revision
        if (snapshot / "config.json").exists():
            return str(snapshot)

    if snapshots.exists():
        for snapshot in sorted(snapshots.iterdir()):
            if (snapshot / "config.json").exists():
                return str(snapshot)

    return path


def resolve_torch_dtype(torch_module: Any, torch_dtype: str) -> Any:
    if torch_dtype == "auto":
        return "auto"
    if not hasattr(torch_module, torch_dtype):
        raise ValueError(f"Unsupported torch dtype: {torch_dtype}")
    return getattr(torch_module, torch_dtype)


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def find_message(record: Dict[str, Any], role: str) -> Optional[Dict[str, Any]]:
    for message in record.get("messages", []):
        if message.get("role") == role:
            return message
    return None


def extract_options(question: str) -> Dict[str, str]:
    options: Dict[str, str] = {}
    for match in re.finditer(r"(?m)^\s*([A-D])\.\s*(.+?)\s*$", question):
        options[match.group(1).upper()] = match.group(2).strip()
    return options


def extract_answer_letter(text: str) -> Optional[str]:
    tail = text[-1000:]
    for pattern in ANSWER_PATTERNS:
        match = pattern.search(tail)
        if match:
            return match.group(1).upper()
    return None


def build_user_prompt(question: str, original_answer: str, answer_letter: Optional[str], option_text: str) -> str:
    answer_hint = (
        f"{answer_letter}. {option_text}" if answer_letter and option_text else answer_letter or "unknown"
    )
    return f"""Question and options:
{question}

Original final answer to preserve:
{answer_hint}

Verbose original answer:
{original_answer}

Rewrite the verbose original answer into the required concise rationale. Preserve the original conclusion."""


def strip_control_text(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return cleaned


def final_answer_pattern(letter: str) -> re.Pattern[str]:
    return re.compile(rf"(?:final\s+answer|answer)\s*(?:is|:)\s*\**\(?{letter}\)?", re.IGNORECASE)


def ensure_final_answer(text: str, answer_letter: Optional[str], option_text: str) -> str:
    if not answer_letter:
        return text

    final_line = f"Final answer: {answer_letter}."
    if option_text:
        final_line += f" {option_text}"

    tail = text[-300:]
    if final_answer_pattern(answer_letter).search(tail):
        return text
    return f"{text.rstrip()} {final_line}"


class QwenRewriter:
    def __init__(
        self,
        model: str,
        max_new_tokens: int,
        batch_size: int,
        temperature: float,
        top_p: float,
        torch_dtype: str,
        device_map: str,
        local_files_only: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved_model = resolve_hf_model_path(model)
        dtype = resolve_torch_dtype(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_model,
            local_files_only=local_files_only,
            trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": local_files_only,
            "trust_remote_code": True,
        }
        if device_map != "none":
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(resolved_model, **model_kwargs)
        if device_map == "none":
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
        self.model.eval()

        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.temperature = temperature
        self.top_p = top_p

    def rewrite_batch(self, prompts: Sequence[str]) -> List[str]:
        import torch

        texts = []
        for prompt in prompts:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            texts.append(text)

        model_inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)
        input_length = model_inputs.input_ids.shape[1]
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0.0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.temperature > 0.0:
            generation_kwargs["temperature"] = self.temperature
            generation_kwargs["top_p"] = self.top_p

        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, **generation_kwargs)

        outputs = []
        for sequence in generated_ids:
            generated = sequence[input_length:]
            outputs.append(strip_control_text(self.tokenizer.decode(generated, skip_special_tokens=True)))
        return outputs


def flush_batch(
    batch: List[Tuple[Dict[str, Any], Dict[str, Any], Optional[str], str, str]],
    rewriter: QwenRewriter,
    output_handle: Any,
    keep_original: bool,
) -> int:
    prompts = [item[4] for item in batch]
    rewrites = rewriter.rewrite_batch(prompts)

    for (record, assistant_message, answer_letter, option_text, _), rewritten in zip(batch, rewrites):
        original_content = assistant_message.get("content", "")
        rewritten = ensure_final_answer(rewritten, answer_letter, option_text)
        if keep_original:
            record["original_assistant_content"] = original_content
        assistant_message["content"] = rewritten
        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_handle.flush()
    return len(batch)


def rewrite_file(args: argparse.Namespace) -> None:
    rewriter = QwenRewriter(
        model=args.model,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        temperature=args.temperature,
        top_p=args.top_p,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        local_files_only=args.local_files_only,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    skipped = 0
    batch: List[Tuple[Dict[str, Any], Dict[str, Any], Optional[str], str, str]] = []

    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for index, (line_no, record) in enumerate(iter_jsonl(args.input_jsonl)):
            if index < args.start:
                continue
            if args.limit is not None and processed >= args.limit:
                break

            user_message = find_message(record, "user")
            assistant_message = find_message(record, "assistant")
            if user_message is None or assistant_message is None:
                skipped += 1
                if args.write_skipped:
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            question = str(user_message.get("content", ""))
            original_answer = str(assistant_message.get("content", ""))
            options = extract_options(question)
            answer_letter = extract_answer_letter(original_answer)
            option_text = options.get(answer_letter or "", "")
            if answer_letter is None:
                print(f"warning: no final answer letter found at line {line_no}", file=sys.stderr)

            prompt = build_user_prompt(question, original_answer, answer_letter, option_text)
            batch.append((record, assistant_message, answer_letter, option_text, prompt))
            processed += 1

            if len(batch) >= args.batch_size:
                done = flush_batch(batch, rewriter, output_handle, args.keep_original)
                print(f"processed {done} items; total {processed}", file=sys.stderr)
                batch.clear()

        if batch:
            done = flush_batch(batch, rewriter, output_handle, args.keep_original)
            print(f"processed {done} items; total {processed}", file=sys.stderr)

    print(f"wrote {processed} rewritten records to {args.output_jsonl}", file=sys.stderr)
    if skipped:
        print(f"skipped {skipped} records without user/assistant messages", file=sys.stderr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite AudioMCQ Gemini CoT rationales into concise evidence-based rationales with Qwen3-8B."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Qwen3-8B Hugging Face ID or local/cache path.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--torch-dtype", default="bfloat16", help="auto, float16, bfloat16, or float32.")
    parser.add_argument("--device-map", default="auto", help='Use "none" to place the whole model on one device.')
    parser.add_argument("--start", type=int, default=0, help="Zero-based input record index to start from.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of records to rewrite.")
    parser.add_argument("--local-files-only", action="store_true", help="Do not download missing model files.")
    parser.add_argument("--keep-original", action="store_true", help="Store the original assistant text in original_assistant_content.")
    parser.add_argument("--write-skipped", action="store_true", help="Write malformed skipped records to the output unchanged.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rewrite_file(args)


if __name__ == "__main__":
    main()
