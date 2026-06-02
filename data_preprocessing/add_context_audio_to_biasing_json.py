import argparse
import json
import random
from pathlib import Path


DEFAULT_INPUT_JSON = "salmonn_data_v1.1/contextual_biasing/ASR_gigaspeech_contextual_ASR_biasing_le10_top20.json"
DEFAULT_METADATA_JSON = "gigaspeech_biasing_audio/metadata_all.json"
DEFAULT_OUTPUT_JSON = "salmonn_data_v1.1/contextual_biasing/ASR_gigaspeech_contextual_ASR_biasing_le10_top20_with_ctx_audios.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Attach one sampled context pronunciation audio to each biasing word."
    )
    parser.add_argument("--input_json", default=DEFAULT_INPUT_JSON)
    parser.add_argument("--metadata_json", default=DEFAULT_METADATA_JSON)
    parser.add_argument("--output_json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-word-policy",
        choices=["error", "drop_sample", "replace_word"],
        default="error",
        help=(
            "error: fail if any biasing word has no audio; "
            "drop_sample: skip samples containing missing words; "
            "replace_word: replace missing words with metadata-covered words and update biasing_list."
        ),
    )
    parser.add_argument(
        "--check-audio-files",
        action="store_true",
        help="Verify every sampled ctx_audio path exists on disk.",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation. Use a negative value for compact JSON.",
    )
    return parser.parse_args()


def detect_top_level_key(obj):
    if "data" in obj and isinstance(obj["data"], list):
        return "data"
    if "annotation" in obj and isinstance(obj["annotation"], list):
        return "annotation"
    raise ValueError("Unsupported JSON format: expected top-level 'data' or 'annotation' list.")


def normalize_metadata(metadata):
    normalized = {}
    for word, paths in metadata.items():
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list):
            paths = []
        normalized[str(word)] = [str(path) for path in paths if str(path)]
    return normalized


def get_missing_words(items, metadata):
    missing = set()
    for sample in items:
        for word in sample.get("biasing_list", []):
            word = str(word)
            if not metadata.get(word):
                missing.add(word)
    return sorted(missing)


def sample_replacement_word(available_words, rng, used_words):
    if len(used_words) < len(available_words):
        while True:
            word = rng.choice(available_words)
            if word not in used_words:
                return word
    return rng.choice(available_words)


def add_ctx_audios(items, metadata, rng, missing_word_policy, check_audio_files, root_dir):
    output_items = []
    dropped = 0
    replaced = 0
    sampled_audio_count = 0
    missing_words_seen = set()
    missing_files = []
    available_words = [word for word, paths in metadata.items() if paths]
    if not available_words:
        raise ValueError("No words with pronunciation audio are available in metadata.")

    for sample in items:
        biasing_list = sample.get("biasing_list", [])
        if not isinstance(biasing_list, list):
            raise ValueError(f"Expected biasing_list to be a list, got {type(biasing_list).__name__}.")

        biasing_words = [str(word) for word in biasing_list]
        sample_missing_words = [word for word in biasing_words if not metadata.get(word)]
        if sample_missing_words:
            missing_words_seen.update(sample_missing_words)
            if missing_word_policy == "drop_sample":
                dropped += 1
                continue
            if missing_word_policy == "replace_word":
                used_words = {word for word in biasing_words if metadata.get(word)}
                biasing_words = list(biasing_words)
                for idx, word in enumerate(biasing_words):
                    if metadata.get(word):
                        continue
                    replacement_word = sample_replacement_word(available_words, rng, used_words)
                    biasing_words[idx] = replacement_word
                    used_words.add(replacement_word)
                    replaced += 1
            else:
                raise ValueError(
                    "Found biasing words without pronunciation audio: "
                    + ", ".join(sorted(set(sample_missing_words))[:20])
                )

        unresolved_words = [word for word in biasing_words if not metadata.get(word)]
        if unresolved_words:
            raise ValueError(
                "Found unresolved biasing words without pronunciation audio after applying policy: "
                + ", ".join(sorted(set(unresolved_words))[:20])
            )

        new_sample = dict(sample)
        new_sample["biasing_list"] = biasing_words
        ctx_audios = []
        for word in biasing_words:
            ctx_audio = rng.choice(metadata[word])
            if check_audio_files and not (root_dir / ctx_audio).exists():
                missing_files.append(ctx_audio)
            ctx_audios.append(ctx_audio)
            sampled_audio_count += 1
        new_sample["ctx_audios"] = ctx_audios
        output_items.append(new_sample)

    if missing_files:
        raise FileNotFoundError(
            "Some sampled ctx_audio files do not exist. Examples: "
            + ", ".join(missing_files[:20])
        )

    return output_items, {
        "input_items": len(items),
        "output_items": len(output_items),
        "dropped_items": dropped,
        "replaced_words": replaced,
        "sampled_audio_count": sampled_audio_count,
        "missing_word_count": len(missing_words_seen),
        "missing_words": sorted(missing_words_seen),
    }


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    metadata_json = Path(args.metadata_json)
    output_json = Path(args.output_json)
    root_dir = Path.cwd()

    with input_json.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    with metadata_json.open("r", encoding="utf-8") as handle:
        metadata = normalize_metadata(json.load(handle))

    top_level_key = detect_top_level_key(obj)
    items = obj[top_level_key]
    missing_words = get_missing_words(items, metadata)
    if missing_words and args.missing_word_policy == "error":
        raise ValueError(
            f"Found {len(missing_words)} biasing words without pronunciation audio. "
            f"Examples: {missing_words[:20]}. "
            "Regenerate those words or rerun with --missing-word-policy drop_sample/replace_word."
        )

    rng = random.Random(args.seed)
    output_items, summary = add_ctx_audios(
        items,
        metadata,
        rng,
        args.missing_word_policy,
        args.check_audio_files,
        root_dir,
    )

    output_obj = dict(obj)
    output_obj[top_level_key] = output_items
    output_json.parent.mkdir(parents=True, exist_ok=True)
    indent = None if args.indent < 0 else args.indent
    with output_json.open("w", encoding="utf-8") as handle:
        json.dump(output_obj, handle, ensure_ascii=False, indent=indent)

    summary.update(
        {
            "input_json": str(input_json),
            "metadata_json": str(metadata_json),
            "output_json": str(output_json),
            "seed": args.seed,
            "missing_word_policy": args.missing_word_policy,
        }
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()