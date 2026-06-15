#!/usr/bin/env python3
"""Restore punctuation in Lhotse cut JSONL(.gz) files with FunASR ct-punc."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import string
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from funasr import AutoModel
from funasr.models.ct_transformer.utils import split_words
from funasr.train_utils.device_funcs import to_device
from tqdm import tqdm


CT_PUNC_MODELSCOPE_ID = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
PUNCTUATION_CHARS = (
    string.punctuation
    + "。？！，、；：“”‘’（）《》【】…—～·「」『』〔〕｛｝￥"
)
PUNCTUATION_TABLE = str.maketrans("", "", PUNCTUATION_CHARS)


def open_text(path: Path, mode: str):
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")


def default_output_path(input_path: Path) -> Path:
    name = input_path.name
    if name.endswith(".jsonl.gz"):
        return input_path.with_name(name.removesuffix(".jsonl.gz") + ".punc.jsonl.gz")
    if name.endswith(".jsonl"):
        return input_path.with_name(name.removesuffix(".jsonl") + ".punc.jsonl")
    return input_path.with_suffix(input_path.suffix + ".punc")


def count_lines(path: Path) -> int:
    with open_text(path, "rt") as f:
        return sum(1 for _ in f)


def depunctuate(text: str) -> str:
    return text.translate(PUNCTUATION_TABLE)


def extract_text_jobs(records: list[dict[str, Any]]) -> tuple[list[str], list[tuple[int, int]]]:
    texts: list[str] = []
    locations: list[tuple[int, int]] = []
    for record_idx, record in enumerate(records):
        for supervision_idx, supervision in enumerate(record.get("supervisions", [])):
            text = supervision.get("text")
            if isinstance(text, str) and text:
                texts.append(text)
                locations.append((record_idx, supervision_idx))
    return texts, locations


def result_text(result: Any) -> str:
    if isinstance(result, dict):
        text = result.get("text", "")
        return text if isinstance(text, str) else str(text)
    return str(result)


def build_punctuated_text(tokens: list[str], punctuations: list[int], punc_list: list[str]) -> str:
    words_with_punc: list[str] = []
    for i, token in enumerate(tokens):
        if (
            i == 0
            or punc_list[punctuations[i - 1]] == "。"
            or punc_list[punctuations[i - 1]] == "？"
        ) and len(token[0].encode()) == 1:
            token = token.capitalize()
        if i == 0 and len(token[0].encode()) == 1:
            token = " " + token
        if i > 0:
            if len(token[0].encode()) == 1 and len(tokens[i - 1][0].encode()) == 1:
                token = " " + token

        words_with_punc.append(token)
        if punc_list[punctuations[i]] != "_":
            punc = punc_list[punctuations[i]]
            if len(token[0].encode()) == 1:
                if punc == "，":
                    punc = ","
                elif punc == "。":
                    punc = "."
                elif punc == "？":
                    punc = "?"
            words_with_punc.append(punc)

    punctuated = "".join(words_with_punc)
    if not punctuated:
        return punctuated
    if punctuated[-1] in ("，", "、"):
        return punctuated[:-1] + "。"
    if punctuated[-1] == ",":
        return punctuated[:-1] + "."
    if punctuated[-1] not in ("。", "？") and len(punctuated[-1].encode()) != 1:
        return punctuated + "。"
    if punctuated[-1] not in (".", "?") and len(punctuated[-1].encode()) == 1:
        return punctuated + "."
    return punctuated


def generate_ct_punc_batch(model: AutoModel, texts: list[str], batch_size: int) -> list[str]:
    ct_model = getattr(model, "model", None)
    tokenizer = getattr(model, "kwargs", {}).get("tokenizer")
    device = getattr(model, "kwargs", {}).get("device", "cpu")
    if not (hasattr(ct_model, "punc_forward") and hasattr(ct_model, "punc_list") and tokenizer):
        raise TypeError("AutoModel does not expose CT-Transformer batch internals.")

    outputs: list[str] = []
    for batch_start in range(0, len(texts), batch_size):
        text_batch = texts[batch_start : batch_start + batch_size]
        token_batch = [
            split_words(text, jieba_usr_dict=getattr(ct_model, "jieba_usr_dict", None))
            for text in text_batch
        ]
        ids_batch = [np.asarray(tokenizer.encode(tokens), dtype="int64") for tokens in token_batch]
        lengths = [len(ids) for ids in ids_batch]
        max_len = max(lengths, default=0)
        if max_len == 0:
            outputs.extend([""] * len(text_batch))
            continue

        padded = np.zeros((len(ids_batch), max_len), dtype="int64")
        for i, ids in enumerate(ids_batch):
            if len(ids):
                padded[i, : len(ids)] = ids

        data = {
            "text": torch.from_numpy(padded),
            "text_lengths": torch.from_numpy(np.asarray(lengths, dtype="int32")),
        }
        data = to_device(data, device)
        with torch.no_grad():
            y, _ = ct_model.punc_forward(**data)
            predictions = y.argmax(dim=-1).detach().cpu().numpy()

        for tokens, length, prediction in zip(token_batch, lengths, predictions):
            punctuations = [int(x) for x in prediction[:length]]
            outputs.append(build_punctuated_text(tokens, punctuations, ct_model.punc_list))

    return outputs


def generate_with_funasr_wrapper(model: AutoModel, texts: list[str], batch_size: int) -> list[str]:
    try:
        return [result_text(result) for result in model.generate(input=texts, batch_size=batch_size)]
    except AssertionError:
        results = []
        for text in texts:
            results.extend(model.generate(input=[text], batch_size=1))
        return [result_text(result) for result in results]


def punctuate_batch(
    model: AutoModel,
    records: list[dict[str, Any]],
    batch_size: int,
) -> tuple[int, int, int]:
    texts, locations = extract_text_jobs(records)
    if not texts:
        return 0, 0, 0

    try:
        candidates = generate_ct_punc_batch(model, texts, batch_size)
    except Exception as exc:
        tqdm.write(f"Direct batched CT-punc failed ({exc}); using FunASR wrapper fallback.")
        candidates = generate_with_funasr_wrapper(model, texts, batch_size)

    if len(candidates) != len(texts):
        return 0, len(texts), len(texts)

    accepted = 0
    rejected = 0
    for original, candidate, (record_idx, supervision_idx) in zip(texts, candidates, locations):
        if depunctuate(candidate) == original:
            records[record_idx]["supervisions"][supervision_idx]["text"] = candidate
            accepted += 1
        else:
            rejected += 1
    return accepted, rejected, len(texts)


def write_records(out_f, records: list[dict[str, Any]]) -> None:
    for record in records:
        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore punctuation in supervisions[*].text using FunASR ct-punc. "
            "A punctuated hypothesis is adopted only when removing punctuation "
            "reconstructs the original text exactly."
        )
    )
    parser.add_argument("input", type=Path, help="Input .jsonl or .jsonl.gz cuts file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to INPUT with .punc before .jsonl(.gz).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Number of utterances per FunASR batch.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Inference device: "auto", "cuda:0", "cpu", etc. Default: auto.',
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=None,
        help="Number of GPUs passed to FunASR. Defaults to 1 for CUDA, otherwise 0.",
    )
    parser.add_argument(
        "--flush-records",
        type=int,
        default=4096,
        help="Read this many records before running punctuation batches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N records, useful for smoke tests.",
    )
    parser.add_argument(
        "--no-count",
        action="store_true",
        help="Skip the initial line count pass. The progress bar will omit the total.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Writable cache directory for FunASR/ModelScope model files.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cuda:0"


def resolve_model_arg(cache_dir: Path | None) -> str:
    if cache_dir is None:
        return "ct-punc"

    local_model_dir = cache_dir / "models" / Path(CT_PUNC_MODELSCOPE_ID)
    if (local_model_dir / "model.pt").exists():
        return str(local_model_dir)
    return "ct-punc"


def main() -> None:
    args = parse_args()
    input_path = args.input
    output_path = args.output or default_output_path(input_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must be different from input path.")

    device = resolve_device(args.device)
    ngpu = args.ngpu if args.ngpu is not None else (1 if device.startswith("cuda") else 0)
    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MODELSCOPE_CACHE"] = str(args.cache_dir)
        os.environ["HF_HOME"] = str(args.cache_dir)

    total = None if args.no_count else count_lines(input_path)
    if args.limit is not None:
        total = min(total, args.limit) if total is not None else args.limit

    model = AutoModel(
        model=resolve_model_arg(args.cache_dir),
        device=device,
        ngpu=ngpu,
        batch_size=args.batch_size,
        disable_pbar=True,
        disable_update=True,
    )

    accepted = 0
    rejected = 0
    total_texts = 0
    processed_records = 0
    started = time.perf_counter()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pending: list[dict[str, Any]] = []
    with open_text(input_path, "rt") as in_f, open_text(output_path, "wt") as out_f:
        pbar = tqdm(total=total, unit="cut", dynamic_ncols=True, desc="punctuating")
        try:
            for line in in_f:
                if args.limit is not None and processed_records >= args.limit:
                    break
                pending.append(json.loads(line))
                processed_records += 1

                if len(pending) >= args.flush_records:
                    ok, bad, seen = punctuate_batch(model, pending, args.batch_size)
                    accepted += ok
                    rejected += bad
                    total_texts += seen
                    write_records(out_f, pending)
                    pending.clear()
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    pbar.update(args.flush_records)
                    pbar.set_postfix(
                        accepted=accepted,
                        rejected=rejected,
                        texts=total_texts,
                        speed=f"{processed_records / elapsed:.1f} cuts/s",
                    )

            if pending:
                ok, bad, seen = punctuate_batch(model, pending, args.batch_size)
                accepted += ok
                rejected += bad
                total_texts += seen
                write_records(out_f, pending)
                pbar.update(len(pending))
                elapsed = max(time.perf_counter() - started, 1e-9)
                pbar.set_postfix(
                    accepted=accepted,
                    rejected=rejected,
                    texts=total_texts,
                    speed=f"{processed_records / elapsed:.1f} cuts/s",
                )
        finally:
            pbar.close()

    print(
        "Done: "
        f"{processed_records} cuts, {total_texts} texts, "
        f"{accepted} accepted, {rejected} rejected -> {output_path}"
    )


if __name__ == "__main__":
    main()
