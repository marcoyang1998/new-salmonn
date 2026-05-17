from ast import parse
import os
os.environ["HF_HUB_OFFLINE"] = "1"   # use local cache; skip network checks
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from fense.evaluator import Evaluator


def load_cands_df(cands_dir, dataset='audiocaps'):
    """Load candidates from either a CSV or JSONL file into a DataFrame.

    For audiocaps: columns are (youtube_id, caption).
    For clotho:    columns are (file_name, caption).
    """
    cands_path = Path(cands_dir)
    if cands_path.suffix == '.jsonl':
        seen = {}
        with open(cands_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if dataset == 'audiocaps':
                    # Path format: .../Y<youtube_id>.wav — strip the single leading 'Y' prefix
                    key = Path(entry['path']).stem[1:]  # e.g. "Y7fmOlUlwoNg" -> "7fmOlUlwoNg"
                else:  # clotho
                    # Path format: .../evaluation/<file_name>.wav
                    key = Path(entry['path']).name  # e.g. "Santa Motor.wav"
                if key not in seen:
                    seen[key] = entry['response']
        id_col = 'youtube_id' if dataset == 'audiocaps' else 'file_name'
        return pd.DataFrame(list(seen.items()), columns=[id_col, 'caption'])
    else:
        return pd.read_csv(cands_dir)


def get_system_score(evaluator, cands_dir, dataset):
    cands_df = load_cands_df(cands_dir, dataset)
    if dataset == 'audiocaps':
        ref_dir = Path(__file__).parents[0] / 'test_data' / 'audiocaps_test.csv'
        ref_df = pd.read_csv(ref_dir)
        available_ids = set(cands_df["youtube_id"])
        # Build refs only for IDs present in candidates
        id2refs = {}
        for rid, row in ref_df.iterrows():
            id0 = row["youtube_id"]
            if id0 not in available_ids:
                continue
            id2refs.setdefault(id0, []).append(row["caption"])
        cand_id2caption = dict(zip(cands_df["youtube_id"], cands_df["caption"]))
        common_ids = [id0 for id0 in id2refs]  # preserves insertion order
        cands = [cand_id2caption[id0] for id0 in common_ids]
        list_refs = [id2refs[id0] for id0 in common_ids]
        print(f"Evaluating {len(cands)}/{len(ref_df) // 5} available items")
        score = evaluator.corpus_score(cands, list_refs)
    
    elif dataset == 'clotho':
        ref_dir = Path(__file__).parents[0] / 'test_data' / 'clotho_eval.csv'
        ref_df = pd.read_csv(ref_dir)
        available_ids = set(cands_df["file_name"])
        cand_id2caption = dict(zip(cands_df["file_name"], cands_df["caption"]))
        # Filter to items present in candidates
        id2refs = {}
        for _, row in ref_df.iterrows():
            id0 = row["file_name"]
            if id0 not in available_ids:
                continue
            id2refs[id0] = row.iloc[1:].tolist()  # caption_1..caption_5
        common_ids = list(id2refs.keys())
        cands = [cand_id2caption[id0] for id0 in common_ids]
        list_refs = [id2refs[id0] for id0 in common_ids]
        print(f"Evaluating {len(cands)}/{len(ref_df)} available items")
        score = evaluator.corpus_score(cands, list_refs)

    return score

def eval_single():
    print("----Using tiny models----")
    evaluator = Evaluator(device='cpu', sbert_model='paraphrase-MiniLM-L6-v2', echecker_model='echecker_clotho_audiocaps_tiny')

    eval_cap = "An engine in idling and a man is speaking and then"
    ref_cap = "A machine makes stitching sounds while people are talking in the background"

    score, error_prob, penalized_score = evaluator.sentence_score(eval_cap, [ref_cap], return_error_prob=True)

    print("Cand:", eval_cap)
    print("Ref:", ref_cap)
    print(f"SBERT sim: {score:.4f}, Error Prob: {error_prob:.4f}, Penalized score: {penalized_score:.4f}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sbert_model", default="paraphrase-TinyBERT-L6-v2")
    parser.add_argument("--echecker_model", default="echecker_clotho_audiocaps_base", choices=["echecker_clotho_audiocaps_base", "echecker_clotho_audiocaps_tiny"])
    parser.add_argument("--cands_dir", default="./test_data/audiocaps_cands.csv")
    parser.add_argument("--dataset", default="audiocaps", choices=["audiocaps", "clotho"])
    args = parser.parse_args()
    print(args)
    evaluator = Evaluator(device=args.device, sbert_model=args.sbert_model, echecker_model=args.echecker_model)
    score = get_system_score(evaluator, args.cands_dir, args.dataset)
    print(f"Avg FENSE score on {args.dataset}: {score:.5f}")


