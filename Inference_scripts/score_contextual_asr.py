# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import deque
from enum import Enum
from pathlib import Path

import argparse
import importlib
import logging
import json
import os
import re
import sys
import unicodedata


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


CONVERSATIONAL_FILLERS = {
    "uh",
    "uhh",
    "um",
    "eh",
    "mm",
    "hm",
    "ah",
    "huh",
    "ha",
    "er",
    "oof",
    "hee",
    "ach",
    "eee",
    "ew",
}

ADDITIONAL_DIACRITICS = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def reduce_repeated_words(text):
    pattern = "."
    for i in range(1, 50):
        p = pattern * i
        text = re.sub(f"({p})" + r"\1{4,200}", r"\1", text)
    for i in range(50, 100):
        p = pattern * i
        text = re.sub(f"({p})" + r"\1{3,200}", r"\1", text)
    return text


def remove_symbols_and_diacritics(text, keep=""):
    return "".join(
        c
        if c in keep
        else ADDITIONAL_DIACRITICS[c]
        if c in ADDITIONAL_DIACRITICS
        else ""
        if unicodedata.category(c) == "Mn"
        else " "
        if unicodedata.category(c)[0] in "MSP"
        else c
        for c in unicodedata.normalize("NFKD", text)
    )


class FallbackEnglishTextNormalizer:
    """Dependency-light fallback matching the toolkit's main Whisper normalizer steps."""

    def __init__(self):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",
            r"'s got\b": " has got",
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.spelling_mapping = self._load_spelling_mapping()

    def _load_spelling_mapping(self):
        candidates = [
            Path(__file__).resolve().parents[2]
            / "SALMONNv1.1_eval"
            / "src"
            / "english.json",
            Path(os.environ.get("SALMONN_EVAL_ROOT", ""))
            / "src"
            / "english.json",
        ]
        for path in candidates:
            if path.is_file():
                with open(path, "r") as f:
                    return json.load(f)
        return {}

    def __call__(self, text):
        text = text.lower()
        text = re.sub(r"[<\[][^>\]]*[>\]]", "", text)
        text = re.sub(r"\(([^)]+?)\)", "", text)
        text = re.sub(self.ignore_patterns, "", text)
        text = re.sub(r"\s+'", "'", text)

        for pattern, replacement in self.replacers.items():
            text = re.sub(pattern, replacement, text)

        text = re.sub(r"(\d),(\d)", r"\1\2", text)
        text = re.sub(r"\.([^0-9]|$)", r" \1", text)
        text = remove_symbols_and_diacritics(text, keep=".%$¢€£")
        text = " ".join(self.spelling_mapping.get(word, word) for word in text.split())
        text = re.sub(r"[.$¢€£]([^0-9])", r" \1", text)
        text = re.sub(r"([^0-9])%", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def infer_dataset_name(args, records=None):
    if args.dataset:
        return args.dataset.lower()
    path_parts = [getattr(args, "input_jsonl", None), args.refs, args.hyps]
    if records:
        path_parts.extend(record.get("path", "") for record in records)
    path_text = " ".join(str(part) for part in path_parts if part).lower()
    if "gigaspeech" in path_text or "giga_speech" in path_text:
        return "gigaspeech"
    return ""


def load_english_normalizer(normalizer_root=None):
    roots = []
    if normalizer_root:
        roots.append(Path(normalizer_root))
    if os.environ.get("SALMONN_EVAL_ROOT"):
        roots.append(Path(os.environ["SALMONN_EVAL_ROOT"]))
    roots.append(Path(__file__).resolve().parents[2] / "SALMONNv1.1_eval")

    for root in roots:
        if not (root / "src" / "whisper_normalizer.py").is_file():
            continue
        try:
            sys.path.insert(0, str(root))
            try:
                import regex  # noqa: F401
            except ModuleNotFoundError:
                sys.modules["regex"] = re
            module = importlib.import_module("src.whisper_normalizer")
            logger.info("Using EnglishTextNormalizer from %s", root)
            return module.EnglishTextNormalizer()
        except Exception as exc:
            logger.warning("Could not load toolkit normalizer from %s: %s", root, exc)
        finally:
            if sys.path and sys.path[0] == str(root):
                sys.path.pop(0)

    logger.warning("Falling back to built-in ASR normalizer without number expansion.")
    return FallbackEnglishTextNormalizer()


def normalize_and_tokenize(text, normalizer, remove_conversational_fillers=False):
    tokens = normalizer(text).split()
    if remove_conversational_fillers:
        tokens = [word for word in tokens if word not in CONVERSATIONAL_FILLERS]
    return tokens


def prepare_biasing_words(words, normalizer):
    bias_tokens = []
    for word in words:
        if isinstance(word, str):
            bias_tokens.extend(normalize_and_tokenize(word, normalizer))
    return set(bias_tokens)


def load_jsonl_records(path):
    records = []
    with open(path, "r") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            if "ground_truth_biasing_list" not in item:
                raise ValueError(
                    f"Missing ground_truth_biasing_list in {path} at line {line_no}."
                )
            biasing_words = item["ground_truth_biasing_list"]
            if not isinstance(biasing_words, list):
                raise ValueError(
                    f"ground_truth_biasing_list must be a list in {path} at line {line_no}."
                )

            records.append(
                {
                    "uttid": item.get("uttid") or item.get("id") or item.get("path") or str(line_no),
                    "path": item.get("path", ""),
                    "text": item.get("text", ""),
                    "response": item.get("response", ""),
                    "biasing_words": biasing_words,
                }
            )
    logger.info("Loaded %d jsonl records from %s", len(records), path)
    return records


def load_tsv_records(args):
    refs = {}
    with open(args.refs, "r") as f:
        for line in f:
            ary = line.strip().split("\t")
            uttid, ref, biasing_words = ary[0], ary[1], json.loads(ary[2])
            refs[uttid] = {"text": ref, "biasing_words": biasing_words}
    logger.info("Loaded %d reference utts from %s", len(refs), args.refs)

    hyps = {}
    with open(args.hyps, "r") as f:
        for line in f:
            ary = line.strip().split("\t")
            # May have empty hypo
            if len(ary) >= 2:
                uttid, hyp = ary[0], ary[1]
            else:
                uttid, hyp = ary[0], ""
            hyps[uttid] = hyp
    logger.info("Loaded %d hypothesis utts from %s", len(hyps), args.hyps)

    if not args.lenient:
        for uttid in refs:
            if uttid in hyps:
                continue
            raise ValueError(
                f"{uttid} missing in hyps! Set `--lenient` flag to ignore this error."
            )

    records = []
    for uttid, ref_info in refs.items():
        if uttid not in hyps:
            continue
        records.append(
            {
                "uttid": uttid,
                "path": uttid,
                "text": ref_info["text"],
                "response": hyps[uttid],
                "biasing_words": ref_info["biasing_words"],
            }
        )
    return records


class Code(Enum):
    match = 1
    substitution = 2
    insertion = 3
    deletion = 4


class AlignmentResult(object):
    def __init__(self, refs, hyps, codes, score):
        self.refs = refs  # deque<int>
        self.hyps = hyps  # deque<int>
        self.codes = codes  # deque<Code>
        self.score = score  # float


class WordError(object):
    def __init__(self):
        self.errors = {
            Code.substitution: 0,
            Code.insertion: 0,
            Code.deletion: 0,
        }
        self.ref_words = 0

    def get_wer(self):
        errors = (
            self.errors[Code.substitution]
            + self.errors[Code.insertion]
            + self.errors[Code.deletion]
        )
        if self.ref_words == 0:
            return float("inf") if errors > 0 else 0.0
        return 100.0 * errors / self.ref_words

    def get_result_string(self):
        return (
            f"error_rate={self.get_wer()}, "
            f"ref_words={self.ref_words}, "
            f"subs={self.errors[Code.substitution]}, "
            f"ins={self.errors[Code.insertion]}, "
            f"dels={self.errors[Code.deletion]}"
        )


def coordinate_to_offset(row, col, ncols):
    return int(row * ncols + col)


def offset_to_row(offset, ncols):
    return int(offset / ncols)


def offset_to_col(offset, ncols):
    return int(offset % ncols)


class EditDistance(object):
    def __init__(self):
        self.scores_ = None
        self.backtraces_ = None
        self.confusion_pairs_ = {}
        self.inserted_words_ = {}
        self.deleted_words_ = {}

    def cost(self, ref, hyp, code):
        if code == Code.match:
            return 0
        elif code == Code.insertion or code == Code.deletion:
            return 3
        else:  # substitution
            return 4

    def get_result(self, refs, hyps):
        res = AlignmentResult(refs=deque(), hyps=deque(), codes=deque(), score=None)

        num_rows, num_cols = len(self.scores_), len(self.scores_[0])
        res.score = self.scores_[num_rows - 1][num_cols - 1]

        curr_offset = coordinate_to_offset(num_rows - 1, num_cols - 1, num_cols)

        while curr_offset != 0:
            curr_row = offset_to_row(curr_offset, num_cols)
            curr_col = offset_to_col(curr_offset, num_cols)

            prev_offset = self.backtraces_[curr_row][curr_col]

            prev_row = offset_to_row(prev_offset, num_cols)
            prev_col = offset_to_col(prev_offset, num_cols)

            res.refs.appendleft(curr_row - 1)
            res.hyps.appendleft(curr_col - 1)
            if curr_row - 1 == prev_row and curr_col == prev_col:
                ref_str = refs[res.refs[0]]
                deleted_word = ref_str
                if deleted_word not in self.deleted_words_:
                    self.deleted_words_[deleted_word] = 1
                else:
                    self.deleted_words_[deleted_word] += 1

                res.codes.appendleft(Code.deletion)

            elif curr_row == prev_row and curr_col - 1 == prev_col:
                hyp_str = hyps[res.hyps[0]]
                inserted_word = hyp_str
                if inserted_word not in self.inserted_words_:
                    self.inserted_words_[inserted_word] = 1
                else:
                    self.inserted_words_[inserted_word] += 1

                res.codes.appendleft(Code.insertion)

            else:
                # assert(curr_row - 1 == prev_row and curr_col - 1 == prev_col)
                ref_str = refs[res.refs[0]]
                hyp_str = hyps[res.hyps[0]]

                if ref_str == hyp_str:
                    res.codes.appendleft(Code.match)
                else:
                    res.codes.appendleft(Code.substitution)

                    confusion_pair = "%s -> %s" % (ref_str, hyp_str)
                    if confusion_pair not in self.confusion_pairs_:
                        self.confusion_pairs_[confusion_pair] = 1
                    else:
                        self.confusion_pairs_[confusion_pair] += 1

            curr_offset = prev_offset

        return res

    def align(self, refs, hyps):
        if len(refs) == 0 and len(hyps) == 0:
            raise ValueError("Doesn't support empty ref AND hyp!")

        # NOTE: we're not resetting the values in these matrices because every value
        # will be overridden in the loop below. If this assumption doesn't hold,
        # be sure to set all entries in self.scores_ and self.backtraces_ to 0.
        self.scores_ = [[0.0] * (len(hyps) + 1) for _ in range(len(refs) + 1)]
        self.backtraces_ = [[0] * (len(hyps) + 1) for _ in range(len(refs) + 1)]

        num_rows, num_cols = len(self.scores_), len(self.scores_[0])

        for i in range(num_rows):
            for j in range(num_cols):
                if i == 0 and j == 0:
                    self.scores_[i][j] = 0.0
                    self.backtraces_[i][j] = 0
                    continue

                if i == 0:
                    self.scores_[i][j] = self.scores_[i][j - 1] + self.cost(
                        None, hyps[j - 1], Code.insertion
                    )
                    self.backtraces_[i][j] = coordinate_to_offset(i, j - 1, num_cols)
                    continue

                if j == 0:
                    self.scores_[i][j] = self.scores_[i - 1][j] + self.cost(
                        refs[i - 1], None, Code.deletion
                    )
                    self.backtraces_[i][j] = coordinate_to_offset(i - 1, j, num_cols)
                    continue

                # Below here both i and j are greater than 0
                ref = refs[i - 1]
                hyp = hyps[j - 1]
                best_score = self.scores_[i - 1][j - 1] + (
                    self.cost(ref, hyp, Code.match)
                    if ref == hyp
                    else self.cost(ref, hyp, Code.substitution)
                )

                prev_row = i - 1
                prev_col = j - 1
                ins = self.scores_[i][j - 1] + self.cost(None, hyp, Code.insertion)
                if ins < best_score:
                    best_score = ins
                    prev_row = i
                    prev_col = j - 1

                delt = self.scores_[i - 1][j] + self.cost(ref, None, Code.deletion)
                if delt < best_score:
                    best_score = delt
                    prev_row = i - 1
                    prev_col = j

                self.scores_[i][j] = best_score
                self.backtraces_[i][j] = coordinate_to_offset(
                    prev_row, prev_col, num_cols
                )

        return self.get_result(refs, hyps)


def main(args):
    if args.input_jsonl:
        records = load_jsonl_records(args.input_jsonl)
    else:
        if not args.refs or not args.hyps:
            raise ValueError("Provide either --input-jsonl or both --refs and --hyps.")
        records = load_tsv_records(args)

    dataset_name = infer_dataset_name(args, records)
    remove_conversational_fillers = dataset_name == "gigaspeech"
    normalizer = load_english_normalizer(args.normalizer_root)
    if remove_conversational_fillers:
        logger.info("Using GigaSpeech scoring mode: removing conversational fillers.")

    # Calculate WER, U-WER, and B-WER
    wer = WordError()
    u_wer = WordError()
    b_wer = WordError()
    for record in records:
        ref_tokens = normalize_and_tokenize(
            record["text"],
            normalizer,
            remove_conversational_fillers=remove_conversational_fillers,
        )
        biasing_words = prepare_biasing_words(
            record["biasing_words"],
            normalizer,
        )
        hyp_tokens = normalize_and_tokenize(
            reduce_repeated_words(record["response"]),
            normalizer,
            remove_conversational_fillers=remove_conversational_fillers,
        )
        if not ref_tokens and not hyp_tokens:
            continue
        ed = EditDistance()
        result = ed.align(ref_tokens, hyp_tokens)
        for code, ref_idx, hyp_idx in zip(result.codes, result.refs, result.hyps):
            if code == Code.match:
                wer.ref_words += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                else:
                    u_wer.ref_words += 1
            elif code == Code.substitution:
                wer.ref_words += 1
                wer.errors[Code.substitution] += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.substitution] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.substitution] += 1
            elif code == Code.deletion:
                wer.ref_words += 1
                wer.errors[Code.deletion] += 1
                if ref_tokens[ref_idx] in biasing_words:
                    b_wer.ref_words += 1
                    b_wer.errors[Code.deletion] += 1
                else:
                    u_wer.ref_words += 1
                    u_wer.errors[Code.deletion] += 1
            elif code == Code.insertion:
                wer.errors[Code.insertion] += 1
                if hyp_tokens[hyp_idx] in biasing_words:
                    b_wer.errors[Code.insertion] += 1
                else:
                    u_wer.errors[Code.insertion] += 1

    # Report results
    print(f"WER: {wer.get_result_string()}")
    print(f"U-WER: {u_wer.get_result_string()}")
    print(f"B-WER: {b_wer.get_result_string()}")


if __name__ ==  "__main__":
    desc = "Compute WER, U-WER, and B-WER. Results are output to stdout."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--input-jsonl",
        default=None,
        help="Path to inference result jsonl. Each line should contain text, response, "
        "and ground_truth_biasing_list.",
    )
    parser.add_argument(
        "--refs",
        default=None,
        help="Path to tab-separated reference file. First column is utterance ID. "
        "Second column is reference text. Last column is list of biasing words.",
    )
    parser.add_argument(
        "--hyps",
        default=None,
        help="Path to tab-separated hypothesis file. First column is utterance ID. "
        "Second column is hypothesis text.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="If set, hyps doesn't have to cover all of refs.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Dataset name. Set to gigaspeech to apply the toolkit's extra "
        "conversational-filler removal. If omitted, this is inferred from paths.",
    )
    parser.add_argument(
        "--normalizer-root",
        default=None,
        help="Optional path to SALMONNv1.1_eval. Defaults to $SALMONN_EVAL_ROOT "
        "or a sibling SALMONNv1.1_eval checkout when available.",
    )
    args = parser.parse_args()
    main(args)
