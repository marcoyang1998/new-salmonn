import argparse
import json
import os
import re
import warnings

import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer

from inference_mmar import (
    ZIPFORMER_LIKE,
    get_fbank,
    parse_model_output,
    prepare_model_inputs,
    str2bool,
)
from inference_utils import (
    ModelArguments,
    maybe_init_qwen3_embedding_model,
    override_args_from_config,
)
from modeling_salmonn import SALMONN
from salmonn_datasets import PETRELOSS_CONFIG, load_audio_from_petrel_oss


WORD_TO_SCORE = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run SALMONN inference on QualiSpeech test scores")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--qualispeech_json",
        type=str,
        default="/scratches/mneme/xy316/workspace/new-salmonn/qualispeech_test_score.json",
        help="Path to qualispeech_test_score.json",
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
        default=16,
        help="Maximum number of tokens to generate for each score.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for quick debugging.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    model_args = ModelArguments()
    model_args.model_name_or_path = args.model_name_or_path
    model_args = override_args_from_config(args.model_name_or_path, model_args)

    if args.encoder_type is not None:
        model_args.encoder_type = args.encoder_type
    if args.concat_encoder_features is not None:
        model_args.concat_encoder_features = args.concat_encoder_features

    print(f"Encoder type: {model_args.encoder_type}")

    if model_args.llm_type == "Qwen":
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    elif model_args.llm_type == "Llama":
        tokenizer = AutoTokenizer.from_pretrained(
            "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct"
        )
    else:
        raise ValueError(f"Unknown llm_type: {model_args.llm_type}")

    model = SALMONN.from_pretrained(
        model_args.model_name_or_path,
        config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path, "config.json")),
        model_args=model_args,
        torch_dtype="auto",
        device_map="auto",
    )
    if model_args.inject_temporal_embedding:
        model.register_temporal_tokens(tokenizer)
    if getattr(model_args, "inject_temporal_embedding_nl", False):
        model.register_nl_timestamp_tokenizer(tokenizer)
    maybe_init_qwen3_embedding_model(model, model_args)
    model.eval()

    return model, tokenizer, model_args


def get_items(payload):
    if isinstance(payload, dict) and "data" in payload:
        return payload["data"]
    if isinstance(payload, list):
        return payload
    raise ValueError("Expected QualiSpeech JSON to be a list or a dict containing a 'data' list.")


def get_user_prompt(item):
    for message in item.get("messages", []):
        if message.get("role") == "user":
            return message.get("content", "")
    raise ValueError("Item is missing a user message.")


def get_reference_score(item):
    for message in item.get("messages", []):
        if message.get("role") == "assistant":
            return str(message.get("content", "")).strip()
    return ""


class AudioLoader:
    def __init__(self):
        self.client = None

    def load(self, audio_path):
        if audio_path.startswith("s3://"):
            if self.client is None:
                from petrel_client.client import Client

                self.client = Client(PETRELOSS_CONFIG)
            return load_audio_from_petrel_oss(audio_path, self.client)
        return torchaudio.load(audio_path)


def extract_features(audio_path, fbank, model_args, audio_loader):
    audio_chunk = model_args.audio_chunk * 16000
    enc = model_args.encoder_type

    feature, raw_wavs, audio_nums, split_feature_lens = [], [], [], []

    audio, fs = audio_loader.load(audio_path)
    if fs != 16000:
        audio = torchaudio.functional.resample(audio, fs, 16000)
        fs = 16000

    if enc in ZIPFORMER_LIKE:
        if model_args.split_audio and audio.shape[-1] > audio_chunk:
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            pad_len = (-audio.shape[-1]) % audio_chunk
            if pad_len > 0:
                audio = F.pad(audio, (0, pad_len))
            audio = audio.unfold(-1, audio_chunk, audio_chunk)[0]
            item_fbanks = fbank.extract_batch(audio, sampling_rate=fs)
            if item_fbanks.ndim == 2:
                item_fbanks = item_fbanks.unsqueeze(0)
            audio_nums.append(item_fbanks.size(0))
            for item_fbank in item_fbanks:
                feature.append(item_fbank)
                split_feature_lens.append(feature[-1].size(0))
            split_feature_lens[-1] = max((audio_chunk - pad_len + 159) // 160, 50)
        else:
            feature.append(fbank.extract(audio.squeeze(), sampling_rate=fs))
    elif enc == "dasheng":
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        feature.append(
            fbank(audio, sampling_rate=fs, return_tensors="pt").input_values.squeeze(0).transpose(0, 1)
        )
    elif enc == "dasheng_wavlm":
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        feature.append(audio.squeeze())
    elif enc == "whisper_beats":
        audio = audio[:, : 30 * 16000]
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        raw_wavs.append(audio.squeeze())
        feature.append(fbank(audio.squeeze().numpy(), sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
    elif enc in ("qwenomni", "qwen3omni"):
        audio = audio[:, : 300 * 16000]
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        one_fbank = fbank(
            audio.squeeze().numpy(),
            sampling_rate=fs,
            return_tensors="pt",
            return_attention_mask=True,
        )
        feature.append(one_fbank["input_features"].squeeze())
        raw_wavs.append(one_fbank["attention_mask"].squeeze())
    elif enc == "mimo":
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        audio = torchaudio.functional.resample(audio, fs, 24000)
        spec = fbank(audio[None, :])
        feature.append(torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1))
    else:
        raise ValueError(f"Unknown encoder_type: {enc}")

    return feature, raw_wavs, audio_nums, split_feature_lens


def extract_score(model_output_raw):
    cleaned = str(model_output_raw).strip().lower()
    numeric_match = re.search(r"(?<!\d)([1-5])(?!\d)", cleaned)
    if numeric_match:
        return numeric_match.group(1), False

    word_pattern = r"\b(" + "|".join(WORD_TO_SCORE) + r")\b"
    word_match = re.search(word_pattern, cleaned)
    if word_match:
        return WORD_TO_SCORE[word_match.group(1)], True

    return "", True


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

    with open(args.qualispeech_json, "r") as f:
        payload = json.load(f)
    items = get_items(payload)
    run_items = items[: args.limit] if args.limit is not None else items

    total = len(run_items)
    parsed = 0
    word_fallbacks = 0
    exact_matches = 0
    wrong_format = 0

    for i, item in enumerate(run_items):
        prompt = get_user_prompt(item)
        audios = item.get("audios", [])
        if len(audios) != 1:
            raise ValueError(f"QualiSpeech item {i} should contain exactly one audio path, got {len(audios)}.")
        audio_path = audios[0]

        model_output_raw = run_one_item(
            audio_path,
            prompt,
            fbank,
            model,
            tokenizer,
            model_args,
            args.max_new_tokens,
            audio_loader,
        )
        prediction, used_fallback = extract_score(model_output_raw)
        item["model_prediction"] = prediction

        if prediction:
            parsed += 1
        if used_fallback:
            if prediction:
                word_fallbacks += 1
            else:
                wrong_format += 1
                warnings.warn(
                    f"Could not parse a 1-to-5 score for {audio_path}. Got: {repr(model_output_raw)}",
                    stacklevel=1,
                )

        reference = get_reference_score(item)
        if prediction and reference and prediction == reference:
            exact_matches += 1

        if i % 100 == 0 and i:
            print(
                f"[{i}/{total}] parsed: {parsed}/{i}, exact match: {exact_matches}/{i} "
                f"({exact_matches / i:.2%}), wrong format: {wrong_format}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    accuracy = exact_matches / total if total > 0 else 0.0
    checkpoint_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))

    print("\n===== Inference Summary =====")
    print(f"Checkpoint     : {model_args.model_name_or_path} ({checkpoint_name})")
    print(f"Encoder        : {model_args.encoder_type}")
    print(f"Total          : {total}")
    print(f"Parsed         : {parsed}")
    print(f"Exact matches  : {exact_matches} ({accuracy:.2%})")
    print(f"Word fallbacks : {word_fallbacks}")
    print(f"Wrong format   : {wrong_format}")
    print(f"Output         : {args.output_path}")


if __name__ == "__main__":
    main()
