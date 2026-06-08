import json
import logging
import math
import re
import warnings
import argparse
import os

from modeling_salmonn import SALMONN
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoFeatureExtractor
import soundfile as sf
from inference_utils import (
    ModelArguments,
    OVERRIDE_KEYS,
    add_mc_prompt_style_arg,
    get_mc_prompt_instruction,
    maybe_init_qwen3_embedding_model,
    override_args_from_config,
)


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

QUESTION_TEMPLATE = (
    # "Answer the following multiple-choice question using only the correct option.\n"
    "Listen to the audio and answer the following multiple-choice question.\n"
    "Question: {question}\n"
    "Choices:\n"
    "{choices_str}\n"
    "{instruction}"
)

AF_QUESTION_TEMPLATE = (
    "{question} Choose one among the following options: \n"
    "{choices_str}"
)

ZIPFORMER_LIKE = {"zipformer2", "spear_transformer"}
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SALMONN on MMAR")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--encoder_type", type=str, default=None,
                        help="Audio encoder type. If not set, value from checkpoint config.json is used.")
    parser.add_argument("--mmar_json", type=str, required=True,
                        help="Path to MMAR-meta.json")
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Root directory containing MMAR audio files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write the annotated output JSON")
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument("--use_af_prompt", type=str2bool, default=False,
                        help="Use 'Choose one among the following options' prompt format")
    add_mc_prompt_style_arg(parser)
    return parser.parse_args()


def build_prompt(question: str, choices: list, use_af_prompt: bool = False, mc_prompt_style: str = "neutral") -> str:
    if use_af_prompt:
        choices_lines = "\n".join(
            f"({LETTERS[i]}) {choice}" for i, choice in enumerate(choices)
        )
        return AF_QUESTION_TEMPLATE.format(question=question, choices_str=choices_lines)
    choices_lines = "\n".join(
        f"Option {LETTERS[i]}: {choice}" for i, choice in enumerate(choices)
    )
    return QUESTION_TEMPLATE.format(
        question=question,
        choices_str=choices_lines,
        instruction=get_mc_prompt_instruction(mc_prompt_style),
    )


def get_fbank(model_args):
    enc = model_args.encoder_type
    if enc in ZIPFORMER_LIKE:
        return Fbank(FbankConfig(num_mel_bins=128))
    elif enc == "dasheng":
        return AutoFeatureExtractor.from_pretrained(
            "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng", trust_remote_code=True
        )
    elif enc == "whisper_beats":
        return AutoFeatureExtractor.from_pretrained(
            "/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2"
        )
    elif enc == "qwenomni":
        return AutoFeatureExtractor.from_pretrained(
            "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B"
        )
    elif enc == "qwen3omni":
        return AutoFeatureExtractor.from_pretrained(
            "/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-Omni-30B-A3B-Instruct"
        )
    elif enc == "mimo":
        from torchaudio.transforms import MelSpectrogram
        return MelSpectrogram(
            sample_rate=24000, n_fft=960, hop_length=240, win_length=960, power=1.0, center=True
        )
    else:
        raise ValueError(f"Unknown encoder_type: {enc}")


def extract_features(audio_path, fbank, model_args):
    """Extract fbank features for a single audio file."""
    audio_chunk = model_args.audio_chunk * 16000
    enc = model_args.encoder_type

    feature, raw_wavs, audio_nums, split_feature_lens = [], [], [], []

    audio, fs = torchaudio.load(audio_path)
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
            split_feature_lens[-1] = max(math.ceil((audio_chunk - pad_len) / 160), 50)
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
        audio = audio[:, :30 * 16000]
        if audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        raw_wavs.append(audio.squeeze())
        sf_audio, _ = sf.read(audio_path, frames=30 * 16000)
        if sf_audio.ndim == 2:
            sf_audio = sf_audio[:, 0]
        feature.append(fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
    elif enc in ("qwenomni", "qwen3omni"):
        sf_audio, _ = sf.read(audio_path, frames=300 * 16000)
        if sf_audio.ndim == 2:
            sf_audio = sf_audio[:, 0]
        one_fbank = fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
        feature.append(one_fbank["input_features"].squeeze())
        raw_wavs.append(one_fbank["attention_mask"].squeeze())
    elif enc == "mimo":
        if audio.ndim == 2:
            audio = audio.mean(dim=0)
        audio = torchaudio.functional.resample(audio, fs, 24000)
        spec = fbank(audio[None, :])
        feature.append(torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1))

    return feature, raw_wavs, audio_nums, split_feature_lens


def prepare_model_inputs(text, audio_nums, tokenizer, model, model_args):
    if model_args.llm_type == "Qwen":
        if model_args.split_audio and len(audio_nums) > 0:
            text_for_model = text
            for one_audio_num in audio_nums:
                text_for_model = text_for_model.replace(
                    "<audio>", "<|vision_start|>" * one_audio_num + "<|vision_end|>", 1
                )
            return tokenizer([text_for_model], return_tensors="pt").to(model.device)
        else:
            return tokenizer(
                [text.replace("<audio>", "<|vision_start|><|vision_end|>")], return_tensors="pt"
            ).to(model.device)
    elif model_args.llm_type == "Llama":
        return tokenizer(
            [text.replace("<audio>", "<|reserved_special_token_0|><|reserved_special_token_1|>")],
            return_tensors="pt",
        ).to(model.device)


def parse_model_output(generated_ids, tokenizer):
    """Strip thinking tokens and return the cleaned output string."""
    output_ids = generated_ids[0].tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)  # </think>
    except ValueError:
        index = 0
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    return content.strip().strip(".")


def extract_predicted_letter(model_output_raw, choices):
    """Parse a letter from model output. Returns (letter, is_wrong_format)."""
    single_letter_pattern = re.compile(r"^[A-Za-z]$")
    option_letter_pattern = re.compile(r"^Option\s+([A-Za-z])$", re.IGNORECASE)
    paren_letter_pattern = re.compile(r"^\(([A-Za-z])\)", re.IGNORECASE)

    if single_letter_pattern.fullmatch(model_output_raw):
        return model_output_raw.upper(), False
    m = option_letter_pattern.fullmatch(model_output_raw)
    if m:
        return m.group(1).upper(), False
    m = paren_letter_pattern.match(model_output_raw)
    if m:
        return m.group(1).upper(), False

    # Wrong format — fall back to first letter found, default to 'A' if out of range
    first = re.search(r"[A-Za-z]", model_output_raw)
    if first:
        candidate = first.group(0).upper()
        letter = candidate if ord(candidate) - ord("A") < len(choices) else "A"
    else:
        letter = "A"
    return letter, True


def main():
    args = parse_args()

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

    fbank = get_fbank(model_args)

    with open(args.mmar_json) as f:
        data = json.load(f)

    total_questions = len(data)
    correct_answers = 0
    wrong_format_answers = 0
    enc = model_args.encoder_type

    for i, item in enumerate(data):
        choices = item["choices"]
        prompt = build_prompt(
            item["question"],
            choices,
            use_af_prompt=args.use_af_prompt,
            mc_prompt_style=args.mc_prompt_style,
        )

        rel = item["audio_path"].lstrip("./")
        audio_path = os.path.join(args.audio_root, rel)

        messages = [{"role": "user", "content": "<audio>" + prompt}]
        # For regular model, enable_thinking is False in default decoding False
        # For thinking model, enable_thinking is True
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        feature, raw_wavs, audio_nums, split_feature_lens = extract_features(audio_path, fbank, model_args)

        if enc == "whisper_beats":
            feature_lens = [f.size(0) for f in raw_wavs]
        elif enc in ZIPFORMER_LIKE and model_args.split_audio and audio_nums:
            feature_lens = split_feature_lens
        else:
            feature_lens = [f.size(0) for f in feature]

        feature_t = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(model.device)
        raw_wavs_t = (
            torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
            if raw_wavs else []
        )
        feature_lens_t = torch.tensor(feature_lens, device=model.device)

        model_inputs = prepare_model_inputs(text, audio_nums, tokenizer, model, model_args)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                fbank_feature=feature_t,
                fbank_feature_len=feature_lens_t,
                raw_wavs=raw_wavs_t,
                user_prompts=[prompt],
                max_new_tokens=500,
            )

        model_output_raw = parse_model_output(generated_ids, tokenizer)
        predicted_letter, is_wrong_format = extract_predicted_letter(model_output_raw, choices)

        if is_wrong_format:
            wrong_format_answers += 1
            warnings.warn(
                f"Invalid output format for {audio_path}. Got: {repr(model_output_raw)}",
                stacklevel=1,
            )

        idx = ord(predicted_letter) - ord("A")
        item["model_prediction"] = choices[idx] if 0 <= idx < len(choices) else choices[0]

        ground_truth = item.get("answer", "")
        if ground_truth and item["model_prediction"].strip().lower() == ground_truth.strip().lower():
            correct_answers += 1

        if i % 100 == 0 and i:
            print(
                f"[{i}/{total_questions}] accuracy: {correct_answers}/{i} "
                f"({correct_answers/i:.2%}), wrong format: {wrong_format_answers}"
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    evaluated_total = sum(1 for d in data if d.get("answer", ""))
    accuracy = correct_answers / evaluated_total if evaluated_total > 0 else 0.0
    checkpoint_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))

    print("\n===== Inference Summary =====")
    print(f"Checkpoint : {model_args.model_name_or_path} ({checkpoint_name})")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"Total      : {total_questions}")
    print(f"Evaluated  : {evaluated_total} (with ground truth)")
    print(f"Correct    : {correct_answers} ({accuracy:.2%})")
    print(f"Wrong fmt  : {wrong_format_answers}")
    print(f"Output     : {args.output_path}")


if __name__ == "__main__":
    main()
