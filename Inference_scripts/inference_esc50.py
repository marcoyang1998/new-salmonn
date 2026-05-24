import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modeling_salmonn import SALMONN
from Inference_scripts.inference_mmau import (
    ModelArguments,
    ZIPFORMER_LIKE,
    extract_features,
    get_fbank,
    parse_model_output,
    prepare_model_inputs,
    str2bool,
)
from Inference_scripts.inference_utils import maybe_init_qwen3_embedding_model, override_args_from_config


PROMPT_TEMPLATE = (
    "Identify the primary sound event in this audio clip.\n"
    "Choose exactly one label from the ESC-50 label set below and answer with only that label.\n"
    "Do not add any explanation, punctuation, or extra words.\n"
    "Use the label text exactly as written, including underscores when present.\n"
    "Labels:\n{labels}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SALMONN sound classification on ESC-50.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--esc50_json",
        type=str,
        default="data/esc50_salmonn.json",
        help="Path to the ESC-50 SALMONN-format JSON.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the annotated output JSON.",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=None,
        help="Audio encoder type. If not set, value from checkpoint config.json is used.",
    )
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument(
        "--test_folds",
        type=str,
        default=None,
        help="Optional comma-separated folds to evaluate. If unset, use all ESC-50 folds.",
    )
    parser.add_argument(
        "--cross_validate_5fold",
        action="store_true",
        help="Run 5-fold evaluation using each ESC-50 fold once as test and report averaged results.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples to process. By default, evaluate the full ESC-50 set.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum number of generated tokens.")
    return parser.parse_args()


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_fold_list(text):
    folds = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        folds.append(int(part))
    if not folds:
        raise ValueError("At least one fold must be provided.")
    return sorted(set(folds))


def build_prompt(labels):
    labels_str = "\n".join(labels)
    return PROMPT_TEMPLATE.format(labels=labels_str)


def get_label_list(payload):
    label_map = payload.get("label_map")
    if isinstance(label_map, dict) and label_map:
        return [label_map[str(idx)] for idx in sorted(int(k) for k in label_map.keys())]

    samples = payload.get("data", payload if isinstance(payload, list) else [])
    labels = []
    seen = set()
    for sample in samples:
        label = sample.get("label") or sample.get("category")
        if label is not None and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def split_by_folds(samples, folds):
    folds_set = set(folds)
    return [sample for sample in samples if int(sample.get("fold", -1)) in folds_set]


def summarize_metrics(total, parsed, correct):
    return {
        "num_samples": total,
        "num_parsed_predictions": parsed,
        "parsed_rate": parsed / total,
        "num_correct": correct,
        "accuracy": correct / total,
    }


def average_fold_metrics(fold_summaries):
    return {
        "avg_parsed_rate": sum(summary["parsed_rate"] for summary in fold_summaries) / len(fold_summaries),
        "avg_accuracy": sum(summary["accuracy"] for summary in fold_summaries) / len(fold_summaries),
    }


def evaluate_fold(fold_name, data, prompt, normalized_to_label, tokenizer, model, model_args, fbank, max_new_tokens):
    total = len(data)
    correct = 0
    parsed = 0
    annotated_data = []

    for idx, sample in enumerate(data, start=1):
        item = dict(sample)
        audio_path = item["audios"][0]
        messages = [{"role": "user", "content": "<audio>" + prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        feature_t, feature_lens_t, raw_wavs_t, audio_nums = get_feature_tensors(audio_path, fbank, model, model_args)
        model_inputs = prepare_model_inputs(text, audio_nums, tokenizer, model, model_args)

        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                fbank_feature=feature_t,
                fbank_feature_len=feature_lens_t,
                raw_wavs=raw_wavs_t,
                user_prompts=[prompt],
                max_new_tokens=max_new_tokens,
            )

        raw_output = parse_model_output(generated_ids, tokenizer)
        predicted_label, parse_mode = extract_prediction(raw_output, labels=None, normalized_to_label=normalized_to_label)
        target_label = item.get("label", item.get("category"))
        is_correct = predicted_label == target_label if predicted_label is not None else False

        if predicted_label is not None:
            parsed += 1
        if is_correct:
            correct += 1

        item["model_output_raw"] = raw_output
        item["model_prediction"] = predicted_label
        item["prediction_parse_mode"] = parse_mode
        item["is_correct"] = is_correct
        annotated_data.append(item)

        if idx % 100 == 0 or idx == total:
            print(
                f"[{fold_name} {idx}/{total}] parsed: {parsed}/{idx} ({parsed / idx:.2%}), "
                f"accuracy: {correct}/{idx} ({correct / idx:.2%})"
            )

    return annotated_data, summarize_metrics(total=total, parsed=parsed, correct=correct)


def get_feature_tensors(audio_path, fbank, model, model_args):
    feature, raw_wavs, audio_nums, split_feature_lens = extract_features(audio_path, fbank, model_args)
    enc = model_args.encoder_type

    if enc == "whisper_beats":
        feature_lens = [wav.size(0) for wav in raw_wavs]
    elif enc in ZIPFORMER_LIKE and model_args.split_audio and audio_nums:
        feature_lens = split_feature_lens
    else:
        feature_lens = [item.size(0) for item in feature]

    feature_t = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(model.device)
    raw_wavs_t = (
        torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
        if raw_wavs
        else []
    )
    feature_lens_t = torch.tensor(feature_lens, device=model.device)
    return feature_t, feature_lens_t, raw_wavs_t, audio_nums


def extract_prediction(raw_output, labels, normalized_to_label):
    normalized_output = normalize_label(raw_output)
    if normalized_output in normalized_to_label:
        return normalized_to_label[normalized_output], "exact"

    matched_labels = [
        normalized_to_label[norm_label]
        for norm_label in normalized_to_label
        if norm_label and norm_label in normalized_output
    ]
    matched_labels = list(dict.fromkeys(matched_labels))
    if len(matched_labels) == 1:
        return matched_labels[0], "substring"

    close_matches = difflib.get_close_matches(
        normalized_output,
        list(normalized_to_label.keys()),
        n=1,
        cutoff=0.75,
    )
    if close_matches:
        return normalized_to_label[close_matches[0]], "fuzzy"

    return None, "unparsed"


def main():
    args = parse_args()
    all_folds = [1, 2, 3, 4, 5]

    model_args = ModelArguments()
    model_args.model_name_or_path = args.model_name_or_path
    model_args = override_args_from_config(args.model_name_or_path, model_args)

    if args.encoder_type is not None:
        model_args.encoder_type = args.encoder_type
    if args.concat_encoder_features is not None:
        model_args.concat_encoder_features = args.concat_encoder_features

    if model_args.llm_type == "Qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.llm_type == "Llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct"
        )
    else:
        raise ValueError(f"Unsupported llm_type: {model_args.llm_type}")

    model = SALMONN.from_pretrained(
        model_args.model_name_or_path,
        config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path, "config.json")),
        model_args=model_args,
        torch_dtype="auto",
        device_map="auto",
    )
    maybe_init_qwen3_embedding_model(model, model_args)
    model.eval()
    fbank = get_fbank(model_args)

    with open(args.esc50_json, "r") as f:
        payload = json.load(f)

    all_data = payload["data"] if isinstance(payload, dict) else payload
    selected_test_folds = parse_fold_list(args.test_folds) if args.test_folds is not None else None
    if args.cross_validate_5fold:
        selected_test_folds = all_folds

    labels = get_label_list(payload if isinstance(payload, dict) else {"data": all_data})
    prompt = build_prompt(labels)
    normalized_to_label = {normalize_label(label): label for label in labels}

    if args.cross_validate_5fold:
        evaluation_specs = [{"fold_name": f"fold-{fold}", "test_folds": [fold]} for fold in all_folds]
    else:
        if selected_test_folds is None:
            data = [dict(sample) for sample in all_data]
            if args.limit is not None:
                data = data[: args.limit]
            if not data:
                raise ValueError(f"No samples found in {args.esc50_json}")
            evaluation_specs = [{"fold_name": "single-run", "test_folds": None, "data": data}]
        else:
            evaluation_specs = [{"fold_name": "single-run", "test_folds": selected_test_folds}]

    fold_summaries = []
    annotated_data = []

    for spec in evaluation_specs:
        if "data" in spec:
            fold_data = spec["data"]
        else:
            fold_data = [dict(sample) for sample in split_by_folds(all_data, spec["test_folds"])]
            if args.limit is not None:
                fold_data = fold_data[: args.limit]
            if not fold_data:
                raise ValueError(f"No samples found for folds {spec['test_folds']}.")

        current_annotated, current_summary = evaluate_fold(
            fold_name=spec["fold_name"],
            data=fold_data,
            prompt=prompt,
            normalized_to_label=normalized_to_label,
            tokenizer=tokenizer,
            model=model,
            model_args=model_args,
            fbank=fbank,
            max_new_tokens=args.max_new_tokens,
        )
        current_summary["test_folds"] = spec["test_folds"]
        fold_summaries.append(current_summary)
        annotated_data.extend(current_annotated)

    overall_total = sum(summary["num_samples"] for summary in fold_summaries)
    overall_parsed = sum(summary["num_parsed_predictions"] for summary in fold_summaries)
    overall_correct = sum(summary["num_correct"] for summary in fold_summaries)
    overall_summary = summarize_metrics(overall_total, overall_parsed, overall_correct)
    average_summary = average_fold_metrics(fold_summaries) if args.cross_validate_5fold else None

    output_payload = payload if isinstance(payload, dict) else {"data": annotated_data}
    output_payload["data"] = annotated_data
    output_payload["inference_prompt"] = prompt
    output_payload["inference_summary"] = {
        "checkpoint": args.model_name_or_path,
        "encoder_type": model_args.encoder_type,
        "max_new_tokens": args.max_new_tokens,
        "cross_validate_5fold": args.cross_validate_5fold,
        "test_folds": selected_test_folds if not args.cross_validate_5fold else all_folds,
        **overall_summary,
    }
    output_payload["fold_summaries"] = fold_summaries
    if average_summary is not None:
        output_payload["cross_validation_average"] = average_summary

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n===== ESC-50 Inference Summary =====")
    print(f"Checkpoint : {args.model_name_or_path}")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"5-fold CV  : {args.cross_validate_5fold}")
    print(f"Test folds : {selected_test_folds if not args.cross_validate_5fold else all_folds}")
    print(f"Total      : {overall_total}")
    print(f"Parsed     : {overall_parsed} ({overall_summary['parsed_rate']:.2%})")
    print(f"Correct    : {overall_correct} ({overall_summary['accuracy']:.2%})")
    if average_summary is not None:
        print("----- 5-fold average -----")
        print(f"Avg parsed  : {average_summary['avg_parsed_rate']:.2%}")
        print(f"Avg accuracy: {average_summary['avg_accuracy']:.2%}")
    print(f"Output     : {args.output_path}")


if __name__ == "__main__":
    main()
