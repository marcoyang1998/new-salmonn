"""Evaluate model predictions on VoxParadox.

Adapted from https://huggingface.co/datasets/IHP-Lab/VoxParadox/blob/main/eval.py
Parsing and scoring rules are unchanged; manifests stream and predictions use
a temporary disk-backed index to keep RAM use bounded.

Usage:
    python eval.py --predictions <preds.jsonl>
    python eval.py --predictions <preds.jsonl> --dataset voxparadox.json --report report.json

Predictions file format (JSONL, one JSON object per line):
    {"id": "age_prediction__0", "response": "Elderly adult"}

The `response` field is the raw model output string. The script parses it into
one of the four MCQ choices (A/B/C/D) using letter-extraction heuristics with a
choice-text fallback, then scores it against:

  * GT Accuracy -- match rate against `answer_gt`.
  * Adversarial-Label Agreement (ALA) -- match rate against any string in
    `adversarial_labels` (the transcript-implied labels).

Both metrics are reported per task and overall (macro = micro since each task
has exactly 200 examples).
"""

import argparse
import json
import os
import re
import unicodedata
import sqlite3
import tempfile

from voxparadox_utils import DATASET_ROOT, iter_records
from collections import defaultdict

# ---------- Response parsing ----------

MAX_TAIL_LINES = 2
MAX_TAIL_CHARS = 1000


def _nfkc(s):
    return unicodedata.normalize("NFKC", s or "")


def _norm(s):
    s = _nfkc(str(s)).casefold().replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", s).strip()


def _tail(text):
    text = _nfkc(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    out = "\n".join(lines[-MAX_TAIL_LINES:]) if lines else text.strip()
    return out[-MAX_TAIL_CHARS:] if len(out) > MAX_TAIL_CHARS else out


def get_choices(item):
    return {L: str(item.get(f"choice_{L.lower()}", "")) for L in "ABCD"}


def _parse_choice_letter(response):
    """Extract an A/B/C/D letter from the response tail.

    First tries to match the entire last line as a single letter (possibly
    with brackets/punctuation), e.g. "A", "(B)", "C." -- this is the most
    confident signal. If that fails, scans the tail for any isolated A/B/C/D
    mention not embedded inside a longer word, and returns the LAST such
    occurrence (handles outputs like "I think the answer is B.").
    """
    if not response:
        return None
    tail = _tail(response)
    lines = [ln.strip() for ln in tail.split("\n") if ln.strip()]
    last = lines[-1] if lines else tail.strip()
    m = re.match(r"(?i)^[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?\s*[,.;:!?\-–—]*\s*$", last)
    if m:
        return m.group(1).upper()
    rx = re.compile(r"(?i)(?<![A-Za-z0-9])[\(\[\{]?\s*([ABCD])\s*[\)\]\}]?\s*[,.;:!?\-–—]*\s*(?=$|\s)")
    ms = list(rx.finditer(tail))
    if ms:
        return ms[-1].group(1).upper()
    return None


def _parse_choice_by_content(response, choices):
    """Fallback parser: match by choice text appearing in the response tail.

    Picks the choice whose normalized text appears LATEST in the tail,
    ranked by `rfind` position. This mirrors the matching used to produce
    the paper's reported numbers, including the known caveat that overlapping
    choice text (e.g., "male" inside "female") is decided by position alone.
    """
    if not response:
        return None
    tail_n = _norm(_tail(response))
    tail_n_sp = tail_n.replace("-", " ")
    best_L, best_pos = None, -1
    for L, txt in choices.items():
        txt_n = _norm(txt)
        if not txt_n:
            continue
        txt_n_sp = txt_n.replace("-", " ")
        for t, hay in [(txt_n, tail_n), (txt_n_sp, tail_n_sp)]:
            pos = hay.rfind(t)
            if pos > best_pos:
                best_pos = pos
                best_L = L
    return best_L if best_pos >= 0 else None


def parse_response(response, item):
    """Map a model response to one of A/B/C/D.

    Letter-first: if the response contains an isolated A/B/C/D mention, use
    it. Otherwise fall back to matching the choice text in the response tail.
    This matches the parsing semantics used to produce the paper's reported
    numbers, so results from this script are directly comparable.
    """
    L = _parse_choice_letter(response)
    if L is not None:
        return L
    return _parse_choice_by_content(response, get_choices(item))


# ---------- Label resolution ----------

def text_to_letter(text, choices):
    if not text:
        return None
    if text.upper() in {"A", "B", "C", "D"}:
        return text.upper()
    n = _norm(text)
    for L, t in choices.items():
        if _norm(t) == n:
            return L
    return None


def gt_letter(item):
    return text_to_letter(item.get("answer_gt", ""), get_choices(item))


def adv_letters(item):
    choices = get_choices(item)
    out = set()
    for s in item.get("adversarial_labels", []) or []:
        L = text_to_letter(s, choices)
        if L:
            out.add(L)
    return out


# ---------- Evaluation ----------

def evaluate(dataset, predictions):
    # Match upstream's last-record-wins behavior for duplicate prediction IDs.
    with tempfile.TemporaryDirectory(prefix="voxparadox_eval_") as tempdir:
        with sqlite3.connect(os.path.join(tempdir, "predictions.sqlite")) as db:
            db.execute("PRAGMA cache_size = -2048")
            db.execute("CREATE TABLE predictions (id TEXT PRIMARY KEY, response TEXT)")
            for prediction in predictions:
                db.execute("INSERT OR REPLACE INTO predictions VALUES (?, ?)",
                           (prediction["id"], prediction.get("response", "")))
            db.commit()
            gt_correct = defaultdict(int)
            adv_correct = defaultdict(int)
            total = defaultdict(int)
            missing = 0
            parse_fail = 0
            for item in dataset:
                task = item["task_name"]
                total[task] += 1
                row = db.execute("SELECT response FROM predictions WHERE id = ?", (item["id"],)).fetchone()
                resp = row[0] if row else None
                if resp is None:
                    missing += 1
                    continue
                pred = parse_response(resp, item)
                if pred is None:
                    parse_fail += 1
                    continue
                gt = gt_letter(item)
                adv = adv_letters(item)
                if gt and pred == gt:
                    gt_correct[task] += 1
                if pred in adv:
                    adv_correct[task] += 1
            return dict(gt_correct), dict(adv_correct), dict(total), missing, parse_fail


def main():
    ap = argparse.ArgumentParser(description="Evaluate model predictions on VoxParadox.")
    ap.add_argument("--predictions", required=True, help="Path to predictions JSONL file.")
    ap.add_argument("--dataset", default=os.path.join(DATASET_ROOT, "metadata.jsonl"),
                    help="Path to VoxParadox JSON array or JSONL manifest.")
    ap.add_argument("--report", default=None, help="Optional path to write a JSON report.")
    args = ap.parse_args()

    n_predictions = sum(1 for _ in iter_records(args.predictions))
    gt_c, adv_c, total, missing, parse_fail = evaluate(
        iter_records(args.dataset), iter_records(args.predictions))

    tasks = sorted(total.keys())
    n_total = sum(total.values())
    sum_gt = sum(gt_c.get(t, 0) for t in tasks)
    sum_adv = sum(adv_c.get(t, 0) for t in tasks)

    print(f"VoxParadox Evaluation")
    print(f"  Dataset:     {n_total} examples across {len(tasks)} tasks")
    print(f"  Predictions: {n_predictions} loaded "
          f"(missing: {missing}, parse-failed: {parse_fail})")
    print()
    print(f"{'Task':<32} {'N':>5} {'GT Acc':>9} {'ALA':>9}")
    print("-" * 58)
    for t in tasks:
        n = total[t]
        gt = 100.0 * gt_c.get(t, 0) / n if n else 0
        ala = 100.0 * adv_c.get(t, 0) / n if n else 0
        print(f"{t:<32} {n:>5} {gt:>8.2f}% {ala:>8.2f}%")
    print("-" * 58)
    overall_gt = 100.0 * sum_gt / n_total if n_total else 0
    overall_ala = 100.0 * sum_adv / n_total if n_total else 0
    print(f"{'Overall':<32} {n_total:>5} {overall_gt:>8.2f}% {overall_ala:>8.2f}%")

    if args.report:
        report = {
            "dataset": os.path.abspath(args.dataset),
            "predictions": os.path.abspath(args.predictions),
            "n_examples": n_total,
            "n_predictions": n_predictions,
            "n_missing": missing,
            "n_parse_failed": parse_fail,
            "overall": {"gt_acc": overall_gt, "ala": overall_ala},
            "per_task": {
                t: {
                    "n": total[t],
                    "gt_correct": gt_c.get(t, 0),
                    "adv_correct": adv_c.get(t, 0),
                    "gt_acc": 100.0 * gt_c.get(t, 0) / total[t] if total[t] else 0,
                    "ala": 100.0 * adv_c.get(t, 0) / total[t] if total[t] else 0,
                }
                for t in tasks
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[report] {args.report}")


if __name__ == "__main__":
    main()
