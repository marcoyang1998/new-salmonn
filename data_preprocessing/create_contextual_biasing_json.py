import argparse
import json
import random
import re
from pathlib import Path


WORD_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a contextual biasing JSON from an original dataset and biasing words."
        ,
        epilog=(
            "Examples:\n"
            "  TSV mode (count-filtered vocab):\n"
            "    python create_contextual_biasing_json.py "
            "--input_json data.json "
            "--word_count_tsv word_counts.tsv "
            "--threshold 27 "
            "--biasing_list_length 50 "
            "--output_folder out\n\n"
            "  TXT mode (one biasing word per line):\n"
            "    python create_contextual_biasing_json.py "
            "--input_json data.json "
            "--bias_words_txt bias_words.txt "
            "--biasing_list_length 50 "
            "--output_folder out"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input_json", required=True, help="Path to the original JSON file.")
    vocab_group = parser.add_mutually_exclusive_group(required=True)
    vocab_group.add_argument("--word_count_tsv", help="Path to the word count TSV.")
    vocab_group.add_argument(
        "--bias_words_txt",
        help="Path to a TXT file containing one biasing word per line.",
    )
    parser.add_argument("--threshold", type=int, help="Keep words with count <= threshold (TSV mode only).")
    parser.add_argument(
        "--biasing_list_length",
        type=int,
        required=True,
        help="Target length for each biasing_list.",
    )
    parser.add_argument("--output_folder", required=True, help="Folder for the generated JSON.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for distractor sampling.",
    )
    parser.add_argument(
        "--task_type_value",
        default="contextual_ASR",
        help="Task type value to write when a sample has a task_type field.",
    )
    return parser.parse_args()


def load_bias_vocab(word_count_tsv: Path, threshold: int) -> list[str]:
    bias_vocab = []
    with word_count_tsv.open("r", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                continue
            word, count_str = parts
            if int(count_str) <= threshold:
                bias_vocab.append(word)
    return bias_vocab


def load_bias_vocab_from_txt(bias_words_txt: Path) -> list[str]:
    bias_vocab = []
    seen = set()
    with bias_words_txt.open("r", encoding="utf-8") as handle:
        for line in handle:
            word = line.strip()
            if not word or word in seen:
                continue
            seen.add(word)
            bias_vocab.append(word)
    return bias_vocab


def detect_top_level_key(obj: dict) -> str:
    if "data" in obj and isinstance(obj["data"], list):
        return "data"
    if "annotation" in obj and isinstance(obj["annotation"], list):
        return "annotation"
    raise ValueError("Unsupported JSON format: expected top-level 'data' or 'annotation' list.")


def extract_transcript(sample: dict) -> str:
    if "messages" in sample and isinstance(sample["messages"], list):
        transcript_parts = []
        for msg in sample["messages"]:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str) and content:
                    transcript_parts.append(content)
        return " ".join(transcript_parts)
    text = sample.get("text", "")
    return text if isinstance(text, str) else ""


def get_ground_truth_biasing_list(transcript: str, bias_set: set[str]) -> list[str]:
    seen = set()
    matched = []
    for token in WORD_PATTERN.findall(transcript.lower()):
        if token in bias_set and token not in seen:
            seen.add(token)
            matched.append(token)
    return matched


def build_biasing_list(
    ground_truth_biasing_list: list[str],
    bias_vocab: list[str],
    biasing_list_length: int,
    rng: random.Random,
) -> list[str]:
    if len(ground_truth_biasing_list) > biasing_list_length:
        biasing_list = list(ground_truth_biasing_list)
        rng.shuffle(biasing_list)
        return biasing_list

    ground_truth_set = set(ground_truth_biasing_list)
    needed = biasing_list_length - len(ground_truth_biasing_list)

    if needed > len(bias_vocab) - len(ground_truth_set):
        raise ValueError(
            f"Not enough distractors: need {needed}, but only {len(bias_vocab) - len(ground_truth_set)} available."
        )

    distractors = []
    distractor_set = set()
    while len(distractors) < needed:
        word = rng.choice(bias_vocab)
        if word in ground_truth_set or word in distractor_set:
            continue
        distractors.append(word)
        distractor_set.add(word)
    biasing_list = list(ground_truth_biasing_list) + distractors
    rng.shuffle(biasing_list)
    return biasing_list


def build_output_path(
    input_json: Path,
    output_folder: Path,
    threshold: int | None,
    biasing_list_length: int,
    vocab_source: str,
) -> Path:
    stem = input_json.stem
    if vocab_source == "word_count_tsv":
        output_name = f"{stem}_contextual_ASR_biasing_le{threshold}_top{biasing_list_length}.json"
    else:
        output_name = f"{stem}_contextual_ASR_biasing_from_txt_top{biasing_list_length}.json"
    return output_folder / output_name


def main():
    args = parse_args()

    input_json = Path(args.input_json)

    if args.word_count_tsv:
        if args.threshold is None:
            raise ValueError("--threshold is required when using --word_count_tsv.")
        vocab_source = "word_count_tsv"
        vocab_source_path = Path(args.word_count_tsv)
    else:
        if args.threshold is not None:
            raise ValueError("--threshold is only valid when using --word_count_tsv.")
        vocab_source = "bias_words_txt"
        vocab_source_path = Path(args.bias_words_txt)

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(
        input_json,
        output_folder,
        args.threshold,
        args.biasing_list_length,
        vocab_source,
    )

    rng = random.Random(args.seed)
    if vocab_source == "word_count_tsv":
        bias_vocab = load_bias_vocab(vocab_source_path, args.threshold)
    else:
        bias_vocab = load_bias_vocab_from_txt(vocab_source_path)
    bias_set = set(bias_vocab)

    if len(bias_vocab) < args.biasing_list_length:
        raise ValueError(
            f"Bias vocabulary size {len(bias_vocab)} is smaller than requested biasing_list_length={args.biasing_list_length}."
        )

    with input_json.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)

    top_level_key = detect_top_level_key(obj)
    items = obj[top_level_key]

    output_items = []
    with_gt = 0
    without_gt = 0
    max_gt_len = 0

    for sample in items:
        new_sample = dict(sample)
        transcript = extract_transcript(sample)
        ground_truth_biasing_list = get_ground_truth_biasing_list(transcript, bias_set)
        biasing_list = build_biasing_list(
            ground_truth_biasing_list,
            bias_vocab,
            args.biasing_list_length,
            rng,
        )

        new_sample["ground_truth_biasing_list"] = ground_truth_biasing_list
        new_sample["biasing_list"] = biasing_list
        if "task_type" in new_sample:
            new_sample["task_type"] = args.task_type_value

        if ground_truth_biasing_list:
            with_gt += 1
        else:
            without_gt += 1
        max_gt_len = max(max_gt_len, len(ground_truth_biasing_list))
        output_items.append(new_sample)

    output_obj = {top_level_key: output_items}
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_obj, handle, ensure_ascii=False, indent=2)

    summary = {
        "input_json": str(input_json),
        "vocab_source": vocab_source,
        "vocab_file": str(vocab_source_path),
        "output_json": str(output_path),
        "threshold": args.threshold,
        "biasing_list_length": args.biasing_list_length,
        "bias_vocab_size": len(bias_vocab),
        "total_items": len(output_items),
        "items_with_ground_truth_biasing_words": with_gt,
        "items_without_ground_truth_biasing_words": without_gt,
        "max_ground_truth_biasing_list_length": max_gt_len,
        "seed": args.seed,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()