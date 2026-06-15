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

from funasr import AutoModel
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


def punctuate_batch(
    model: AutoModel,
    records: list[dict[str, Any]],
    batch_size: int,
) -> tuple[int, int, int]:
    texts, locations = extract_text_jobs(records)
    if not texts:
        return 0, 0, 0

    try:
        results = model.generate(input=texts, batch_size=batch_size)
    except AssertionError:
        tqdm.write(
            "FunASR ct-punc rejected multi-text inference; falling back to "
            "single-text calls for this batch."
        )
        results = []
        for text in texts:
            results.extend(model.generate(input=[text], batch_size=1))
    if len(results) != len(texts):
        return 0, len(texts), len(texts)

    accepted = 0
    rejected = 0
    for original, result, (record_idx, supervision_idx) in zip(texts, results, locations):
        candidate = result_text(result)
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
