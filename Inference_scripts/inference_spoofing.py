import argparse
import json
import os

import torch

from inference_mmar import (
    ZIPFORMER_LIKE,
    get_fbank,
    parse_model_output,
    prepare_model_inputs,
    str2bool,
)
from inference_qualispeech import AudioLoader, extract_features, load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Run SALMONN inference on spoofing test data")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--spoofing_json",
        type=str,
        required=True,
        help="Path to spoofing test JSON",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write the annotated output JSON",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=None,
        help="Audio encoder type. If unset, the value from checkpoint config.json is used.",
    )
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate for each prediction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )
    return parser.parse_args()


def get_items(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected spoofing JSON to be a list or a dict containing a 'data' list.")


def get_user_prompt(item):
    for message in item.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    raise ValueError("Item is missing a user message.")


def run_one_item(audio_path, prompt, fbank, model, tokenizer, model_args, max_new_tokens, audio_loader):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    feature, raw_wavs, audio_nums, split_feature_lens = extract_features(
        audio_path,
        fbank,
        model_args,
        audio_loader,
    )
    if model_args.encoder_type == "whisper_beats":
        feature_lens = [f.size(0) for f in raw_wavs]
    elif model_args.encoder_type in ZIPFORMER_LIKE and model_args.split_audio and audio_nums:
        feature_lens = split_feature_lens
    else:
        feature_lens = [f.size(0) for f in feature]

    feature_t = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(model.device)
    raw_wavs_t = (
        torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
        if raw_wavs
        else []
    )
    feature_lens_t = torch.tensor(feature_lens, device=model.device)
    model_inputs = prepare_model_inputs(text, audio_nums, tokenizer, model, model_args)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            fbank_feature=feature_t,
            fbank_feature_len=feature_lens_t,
            raw_wavs=raw_wavs_t,
            user_prompts=[prompt.replace("<audio>", "", 1).strip()],
            max_new_tokens=max_new_tokens,
        )

    return parse_model_output(generated_ids, tokenizer)


def main():
    args = parse_args()
    model, tokenizer, model_args = load_model_and_tokenizer(args)
    fbank = get_fbank(model_args)
    audio_loader = AudioLoader()

    with open(args.spoofing_json, "r") as f:
        payload = json.load(f)
    items = get_items(payload)
    run_items = items[: args.limit] if args.limit is not None else items

    total = len(run_items)
    for i, item in enumerate(run_items):
        prompt = get_user_prompt(item)
        audios = item.get("audios", [])
        if len(audios) != 1:
            raise ValueError(f"Spoofing item {i} should contain exactly one audio path, got {len(audios)}.")

        item["model_prediction"] = run_one_item(
            audios[0],
            prompt,
            fbank,
            model,
            tokenizer,
            model_args,
            args.max_new_tokens,
            audio_loader,
        )

        if i % 100 == 0 and i:
            print(f"[{i}/{total}] processed")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    checkpoint_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))
    print("\n===== Inference Summary =====")
    print(f"Checkpoint : {model_args.model_name_or_path} ({checkpoint_name})")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"Total      : {total}")
    print(f"Output     : {args.output_path}")


if __name__ == "__main__":
    main()
