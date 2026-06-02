#!/usr/bin/env python3
"""
Compute audio durations for a SALMONN-format JSON dataset.

Each item in data["data"] has the shape:
    {"messages": [...], "audios": [...]}

This script reads the header of each audio file (no full decode) via
soundfile.info(), then appends a "durations" field (list of floats, one
per audio path) to every item and saves the result.

Usage:
    python compute_audio_durations.py \
        --input  /path/to/dataset.json \
        --output /path/to/dataset_with_durations.json \
        --workers 32 \
        --chunk-size 2000
"""

import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import soundfile as sf

PETRELOSS_CONFIG = "/mnt/shared-storage-user/housiyuan/xiaoyu/petreloss.conf"

# Module-level petrel client (initialised lazily per worker process)
_petrel_client = None
_mount_path = None  # set by main() before spawning workers


def _get_petrel_client():
    global _petrel_client
    if _petrel_client is None:
        from petrel_client.client import Client
        _petrel_client = Client(PETRELOSS_CONFIG)
    return _petrel_client


def _resolve_path(path: str) -> tuple:
    """Return (resolved_path, use_petrel)."""
    if path.startswith("s3://") and _mount_path is not None:
        # Strip the s3:// prefix and join with mount path
        local_path = os.path.join(_mount_path, path[len("s3://"):])
        return local_path, False
    return path, path.startswith("s3://")


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect C-level stderr to /dev/null to silence library warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


# ---------------------------------------------------------------------------
# Worker functions (must be module-level for pickle)
# ---------------------------------------------------------------------------

def _get_duration(path: str) -> float:
    """Read audio header and return duration in seconds. Returns -1.0 on error."""
    try:
        resolved, use_petrel = _resolve_path(path)
        if resolved.lower().endswith(".mp4"):
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", resolved],
                capture_output=True, text=True, timeout=30
            )
            return float(result.stdout.strip())
        with _suppress_c_stderr():
            if use_petrel:
                client = _get_petrel_client()
                bytes_data = client.get(resolved)
                buf = io.BytesIO(bytes_data)
                info = sf.info(buf)
            else:
                info = sf.info(resolved)
        return info.frames / info.samplerate
    except Exception:
        return -1.0


def _process_chunk(args):
    """Process a list of items; return list of duration lists."""
    items, start_idx = args
    results = []
    errors = []
    for local_idx, item in enumerate(items):
        durations = []
        for path in item.get("audios", []):
            dur = _get_duration(path)
            if dur < 0:
                errors.append((start_idx + local_idx, path))
            durations.append(dur)
        results.append(durations)
    return results, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Append duration statistics to a SALMONN dataset JSON."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input JSON file.",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help=(
            "Path to output JSON file. "
            "Defaults to <input_stem>_with_durations.json in the same directory."
        ),
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=min(32, os.cpu_count() or 8),
        help="Number of worker processes (default: min(32, cpu_count)).",
    )
    parser.add_argument(
        "--chunk-size", "-c",
        type=int,
        default=2000,
        help="Number of items per worker chunk (default: 2000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    parser.add_argument(
        "--mount-path",
        default=None,
        help=(
            "If provided, s3:// paths are resolved as local paths under this mount point "
            "instead of using petrel client. The s3:// prefix is stripped and the remainder "
            "is joined with this directory."
        ),
    )
    parser.add_argument(
        "--broken-list",
        default=None,
        help="Path to save a text file listing all audio paths that could not be read.",
    )
    parser.add_argument(
        "--debug-n",
        type=int,
        default=0,
        metavar="N",
        help="Debug mode: process only the first N items in a single process (no Pool). "
             "Output is printed but not saved. 0 = disabled.",
    )
    return parser.parse_args()


def build_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_with_durations{ext}"


def make_chunks(items, chunk_size):
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size], start


def print_stats(all_durations):
    flat = [d for durations in all_durations for d in durations if d >= 0]
    if not flat:
        print("No valid durations collected.")
        return
    total_h = sum(flat) / 3600
    mean_s = sum(flat) / len(flat)
    min_s = min(flat)
    max_s = max(flat)
    print(f"\n--- Duration Statistics ---")
    print(f"  Items processed : {len(all_durations)}")
    print(f"  Audio files     : {len(flat)}")
    print(f"  Total duration  : {total_h:.2f} h  ({sum(flat):.1f} s)")
    print(f"  Mean            : {mean_s:.3f} s")
    print(f"  Min             : {min_s:.3f} s")
    print(f"  Max             : {max_s:.3f} s")
    # simple histogram
    buckets = [0, 5, 10, 20, 30, 60, 120, 180, float("inf")]
    counts = [0] * (len(buckets) - 1)
    for d in flat:
        for i in range(len(buckets) - 1):
            if buckets[i] <= d < buckets[i + 1]:
                counts[i] += 1
                break
    print(f"\n  Duration histogram:")
    for i, cnt in enumerate(counts):
        lo = buckets[i]
        hi = buckets[i + 1]
        hi_str = f"{hi:.0f}s" if hi != float("inf") else "inf"
        bar = "#" * (cnt * 40 // max(counts))
        print(f"    [{lo:>4.0f}s – {hi_str:>5}): {cnt:>8}  {bar}")
    print("")


def main():
    args = parse_args()

    global _mount_path
    _mount_path = args.mount_path

    # ------------------------------------------------------------------ debug
    if args.debug_n > 0:
        print(f"[DEBUG] Single-process mode — processing first {args.debug_n} items.", flush=True)
        with open(args.input, "r") as f:
            dataset = json.load(f)
        items = dataset["data"][:args.debug_n]
        all_durations_flat = []
        all_errors = []
        for i, item in enumerate(items):
            durations = []
            for path in item.get("audios", []):
                dur = _get_duration(path)
                if dur < 0:
                    all_errors.append((i, path))
                durations.append(dur)
            all_durations_flat.append(durations)
            print(f"  [{i+1}/{args.debug_n}] {item.get('audios', [])} -> {durations}", flush=True)
        if all_errors:
            print(f"\nERRORS ({len(all_errors)}):")
            for idx, path in all_errors:
                print(f"  item {idx}: {path}")
        print_stats(all_durations_flat)
        return
    # -----------------------------------------------------------------------

    output_path = args.output or build_output_path(args.input)

    if os.path.exists(output_path) and not args.overwrite:
        print(
            f"Output file already exists: {output_path}\n"
            "Use --overwrite to replace it."
        )
        sys.exit(1)

    print(f"Input  : {args.input}")
    print(f"Output : {output_path}")
    print(f"Workers: {args.workers}  |  Chunk size: {args.chunk_size}")

    # ---- load ---------------------------------------------------------------
    t0 = time.time()
    print("\nLoading JSON …", flush=True)
    with open(args.input, "r") as f:
        dataset = json.load(f)
    items = dataset["data"]
    print(f"  {len(items):,} items loaded in {time.time() - t0:.1f}s")

    # ---- multiprocessing ----------------------------------------------------
    chunks = list(make_chunks(items, args.chunk_size))
    n_chunks = len(chunks)
    print(f"\nProcessing {n_chunks} chunks with {args.workers} workers …", flush=True)

    all_durations_flat = []
    all_errors = []
    t1 = time.time()

    with Pool(processes=args.workers) as pool:
        for idx, (chunk_results, chunk_errors) in enumerate(
            pool.imap(_process_chunk, chunks, chunksize=1)
        ):
            all_durations_flat.extend(chunk_results)
            all_errors.extend(chunk_errors)
            done = idx + 1
            elapsed = time.time() - t1
            rate = done / elapsed
            eta = (n_chunks - done) / rate if rate > 0 else 0
            print(
                f"\r  Chunk {done}/{n_chunks}  |  "
                f"elapsed {elapsed:.0f}s  |  ETA {eta:.0f}s   ",
                end="",
                flush=True,
            )

    print(f"\n\nFinished reading durations in {time.time() - t1:.1f}s")

    if all_errors:
        print(f"\n--- Skipped Audio Summary ---")
        print(f"  Total skipped : {len(all_errors)}")
        # Break down by path prefix
        from collections import Counter
        def _prefix(path):
            parts = path.replace("s3://", "").split("/")
            return parts[0] if parts else path
        counts = Counter(_prefix(path) for _, path in all_errors)
        for prefix, cnt in counts.most_common():
            print(f"  {prefix:<40} {cnt:>8}")
        print("")

        if args.broken_list:
            with open(args.broken_list, "w") as f:
                for _, path in all_errors:
                    f.write(path + "\n")
            print(f"  Broken audio list saved to: {args.broken_list}")

    # ---- attach durations and filter broken items ---------------------------
    assert len(all_durations_flat) == len(items), "Length mismatch!"
    clean_items = []
    skipped_items = 0
    for item, durations in zip(items, all_durations_flat):
        if any(d < 0 for d in durations):
            skipped_items += 1
        else:
            item["durations"] = durations
            clean_items.append(item)
    dataset["data"] = clean_items
    print(f"  Items removed (broken audio): {skipped_items}")
    print(f"  Items kept                  : {len(clean_items)}")

    # ---- print stats --------------------------------------------------------
    print_stats([d for d in all_durations_flat if all(v >= 0 for v in d)])

    # ---- save ---------------------------------------------------------------
    print(f"Saving to {output_path} …", flush=True)
    t3 = time.time()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"  Done in {time.time() - t3:.1f}s")
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
