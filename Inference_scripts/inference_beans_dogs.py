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
    "Identify which dog produced this bark clip.\n"
    "Choose exactly one label from the BEANS dog identity label set below and answer with only that label.\n"
    "Do not add any explanation, punctuation, or extra words.\n"
    "Use the label text exactly as written.\n"
    "Labels:\n{labels}"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run SALMONN dog identity classification on BEANS dogs.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--beans_dogs_json",
        type=str,
        default="data/beans_dogs_salmonn.json",
        help="Path to the BEANS dogs SALMONN-format JSON.",
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
        "--split",
        type=str,
        choices=("all", "train", "test", "val"),
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of samples to process. By default, evaluate the full selected split.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum number of generated tokens.")
    return parser.parse_args()


def normalize_label(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_prompt(labels):
    return PROMPT_TEMPLATE.format(labels="\n".join(labels))


def get_label_list(payload):
    label_map = payload.get("label_map")
    if isinstance(label_map, dict) and label_map:
        return [label_map[str(idx)] for idx in sorted(int(key) for key in label_map.keys())]

    samples = payload.get("data", payload if isinstance(payload, list) else [])
    labels = []
    seen = set()
    for sample in samples:
        label = sample.get("label") or sample.get("category")
        if label is not None and label not in seen:
            labels.append(label)
            seen.add(label)
    return labels


def select_split(samples, split):
    if split == "all":
        return [dict(sample) for sample in samples]
    return [dict(sample) for sample in samples if sample.get("split") == split]


def summarize_metrics(total, parsed, correct):
    return {
        "num_samples": total,
        "num_parsed_predictions": parsed,
        "parsed_rate": parsed / total,
        "num_correct": correct,
        "accuracy": correct / total,
    }


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


def extract_prediction(raw_output, normalized_to_label):
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


def load_model(model_name_or_path, encoder_type=None, concat_encoder_features=None):
    model_args = ModelArguments()
    model_args.model_name_or_path = model_name_or_path
    model_args = override_args_from_config(model_name_or_path, model_args)

    if encoder_type is not None:
        model_args.encoder_type = encoder_type
    if concat_encoder_features is not None:
        model_args.concat_encoder_features = concat_encoder_features

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
    return model, tokenizer, model_args


def evaluate(data, prompt, normalized_to_label, tokenizer, model, model_args, fbank, max_new_tokens):
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
        predicted_label, parse_mode = extract_prediction(raw_output, normalized_to_label=normalized_to_label)
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
                f"[{idx}/{total}] parsed: {parsed}/{idx} ({parsed / idx:.2%}), "
                f"accuracy: {correct}/{idx} ({correct / idx:.2%})"
            )

    return annotated_data, summarize_metrics(total=total, parsed=parsed, correct=correct)


def main():
    args = parse_args()

    model, tokenizer, model_args = load_model(
        model_name_or_path=args.model_name_or_path,
        encoder_type=args.encoder_type,
        concat_encoder_features=args.concat_encoder_features,
    )
    fbank = get_fbank(model_args)

    with open(args.beans_dogs_json, "r") as f:
        payload = json.load(f)

    all_data = payload["data"] if isinstance(payload, dict) else payload
    data = select_split(all_data, args.split)
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError(f"No samples found in {args.beans_dogs_json} for split={args.split}")

    labels = get_label_list(payload if isinstance(payload, dict) else {"data": all_data})
    prompt = build_prompt(labels)
    normalized_to_label = {normalize_label(label): label for label in labels}

    annotated_data, summary = evaluate(
        data=data,
        prompt=prompt,
        normalized_to_label=normalized_to_label,
        tokenizer=tokenizer,
        model=model,
        model_args=model_args,
        fbank=fbank,
        max_new_tokens=args.max_new_tokens,
    )

    output_payload = dict(payload) if isinstance(payload, dict) else {"data": annotated_data}
    output_payload["data"] = annotated_data
    output_payload["inference_prompt"] = prompt
    output_payload["inference_summary"] = {
        "checkpoint": args.model_name_or_path,
        "encoder_type": model_args.encoder_type,
        "max_new_tokens": args.max_new_tokens,
        "split": args.split,
        **summary,
    }

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n===== BEANS Dogs Inference Summary =====")
    print(f"Checkpoint : {args.model_name_or_path}")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"Split      : {args.split}")
    print(f"Total      : {summary['num_samples']}")
    print(f"Parsed     : {summary['num_parsed_predictions']} ({summary['parsed_rate']:.2%})")
    print(f"Correct    : {summary['num_correct']} ({summary['accuracy']:.2%})")
    print(f"Output     : {args.output_path}")


if __name__ == "__main__":
    main()
