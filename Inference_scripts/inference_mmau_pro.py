import json
import logging
import math
import re
import warnings
import argparse
import os

import pandas as pd
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
# MC_EXCLUDED = {"open", "instruction following", "multi"}
MC_EXCLUDED = set()  # all categories are now handled
# MC_EXCLUDED = {"open", "multi"}

ORDINALS = ["first", "second", "third", "fourth", "fifth"]

QUESTION_TEMPLATE = (
    "Answer the following multiple-choice question using only the correct option.\n"
    "Question: {question}\n"
    "Choices:\n"
    "{choices_str}\n"
    "{instruction}"
)

OPEN_TEMPLATE = "Question: {question}\nPlease answer the question based on the audio."

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
    parser = argparse.ArgumentParser(description="Evaluate SALMONN on MMAU-Pro (multiple choice only)")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--encoder_type", type=str, default=None,
                        help="Audio encoder type. If not set, value from checkpoint config.json is used.")
    parser.add_argument("--mmau_parquet", type=str, required=True,
                        help="Path to MMAU-Pro test.parquet")
    parser.add_argument("--audio_root", type=str, required=True,
                        help="Root directory containing MMAU-Pro audio files")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to write output parquet with model_output column")
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (for debugging)")
    add_mc_prompt_style_arg(parser)
    return parser.parse_args()


def build_prompt(question: str, choices: list, mc_prompt_style: str = "neutral") -> str:
    choices_lines = "\n".join(
        f"Option {LETTERS[i]}: {choice}" for i, choice in enumerate(choices)
    )
    return QUESTION_TEMPLATE.format(
        question=question,
        choices_str=choices_lines,
        instruction=get_mc_prompt_instruction(mc_prompt_style),
    )


def get_correct_letter(answer_text: str, choices: list):
    """Return the option letter (A, B, ...) whose text matches answer_text, or None."""
    for i, c in enumerate(choices):
        if str(c).strip() == str(answer_text).strip():
            return LETTERS[i]
    return None


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

    if single_letter_pattern.fullmatch(model_output_raw):
        letter = model_output_raw.upper()
        if ord(letter) - ord("A") >= len(choices):
            return "A", True
        return letter, False
    m = option_letter_pattern.fullmatch(model_output_raw)
    if m:
        letter = m.group(1).upper()
        if ord(letter) - ord("A") >= len(choices):
            return "A", True
        return letter, False

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
    maybe_init_qwen3_embedding_model(model, model_args)
    model.eval()

    fbank = get_fbank(model_args)

    # Load and filter to MC-only categories
    df = pd.read_parquet(args.mmau_parquet)
    mc_df = df[~df["category"].isin(MC_EXCLUDED)].reset_index(drop=True)
    if args.max_samples is not None:
        mc_df = mc_df.iloc[: args.max_samples].reset_index(drop=True)

    total = len(mc_df)
    print(f"Total MC samples: {total} (excluded categories: {MC_EXCLUDED})")

    # Resume: load existing output and skip already-processed rows
    existing_outputs = {}
    if os.path.exists(args.output_path):
        existing_df = pd.read_parquet(args.output_path)
        if "model_output" in existing_df.columns:
            for _, ex_row in existing_df.iterrows():
                key = (str(ex_row["audio_path"]), str(ex_row["question"]))
                existing_outputs[key] = ex_row["model_output"]
            print(f"Resuming: found {len(existing_outputs)} already-processed samples in {args.output_path}")

    enc = model_args.encoder_type

    # Build model_outputs list in row order; None = not yet evaluated
    model_outputs = []
    new_rows = []
    for i, row in mc_df.iterrows():
        row_key = (str(row["audio_path"]), str(row["question"]))
        if row_key in existing_outputs:
            model_outputs.append(existing_outputs[row_key])
        else:
            model_outputs.append(None)
            new_rows.append((i, row))

    total_new = len(new_rows)
    print(f"New samples to evaluate: {total_new} (skipping {len(mc_df) - total_new} cached)")

    correct_answers = 0
    wrong_format_answers = 0
    mc_steps = 0
    cat_correct = {}
    cat_total = {}

    for step, (i, row) in enumerate(new_rows, 1):
        question = row["question"]
        category = row["category"]
        is_open = category == "open"
        is_multi = category == "multi"
        is_instruction_following = category == "instruction following"

        raw_audio_paths = [os.path.join(args.audio_root, p.lstrip("./")) for p in row["audio_path"]]

        if is_multi:
            # Concatenate all clips into a single waveform
            n = len(raw_audio_paths)
            waves = []
            for p in raw_audio_paths:
                wav, fs = torchaudio.load(p)
                if fs != 16000:
                    wav = torchaudio.functional.resample(wav, fs, 16000)
                if wav.size(0) > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                waves.append(wav)
            combined = torch.cat(waves, dim=1)
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            torchaudio.save(tmp.name, combined, 16000)
            audio_path = tmp.name

            clips = " and ".join(ORDINALS[:n])
            parts = ", ".join(f"the {ORDINALS[j]} part" for j in range(n))
            multi_prefix = (
                f"Listen to the {clips} audio clips and answer the question. "
                f"The audio has been concatenated in order: {parts} correspond "
                f"to the {clips} clips respectively. "
            )
            choices = list(row["choices"])
            answer_text = str(row["answer"])
            correct_letter = get_correct_letter(answer_text, choices)
            prompt = multi_prefix + build_prompt(question, choices, mc_prompt_style=args.mc_prompt_style)
        elif is_open:
            audio_path = raw_audio_paths[0]
            prompt = OPEN_TEMPLATE.format(question=question)
        elif is_instruction_following:
            audio_path = raw_audio_paths[0]
            prompt = "Follow the instruction at the start of an audio."
        else:
            audio_path = raw_audio_paths[0]
            choices = list(row["choices"])
            answer_text = str(row["answer"])
            correct_letter = get_correct_letter(answer_text, choices)
            prompt = build_prompt(question, choices, mc_prompt_style=args.mc_prompt_style)

        messages = [{"role": "user", "content": "<audio>" + prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,  # True necessary for reasoning models
        )

        feature, raw_wavs, audio_nums, split_feature_lens = extract_features(
            audio_path, fbank, model_args
        )

        if enc == "whisper_beats":
            feature_lens = [f.size(0) for f in raw_wavs]
        elif enc in ZIPFORMER_LIKE and model_args.split_audio and audio_nums:
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
                user_prompts=[prompt],
                max_new_tokens=500,
            )

        model_output_raw = parse_model_output(generated_ids, tokenizer)

        if is_multi:
            os.unlink(audio_path)  # remove temp file

        if is_open or is_instruction_following:
            model_outputs[i] = model_output_raw
        else:
            predicted_letter, is_wrong_format = extract_predicted_letter(model_output_raw, choices)
            model_outputs[i] = choices[ord(predicted_letter) - ord("A")]

            if is_wrong_format:
                wrong_format_answers += 1
                warnings.warn(
                    f"Invalid output format for {row['audio_path']}. Got: {repr(model_output_raw)}",
                    stacklevel=1,
                )

            cat_total[category] = cat_total.get(category, 0) + 1
            mc_steps += 1
            if correct_letter is not None and predicted_letter == correct_letter:
                correct_answers += 1
                cat_correct[category] = cat_correct.get(category, 0) + 1

        if step % 100 == 0 or step == total_new:
            acc_str = f"{correct_answers}/{mc_steps} ({correct_answers/mc_steps:.2%})" if mc_steps > 0 else "N/A (no MC items yet)"
            print(f"[{step}/{total_new}] MC acc so far: {acc_str}")

    mc_df = mc_df.copy()
    mc_df["model_output"] = model_outputs

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    mc_df.to_parquet(args.output_path, index=False)

    accuracy = correct_answers / mc_steps if mc_steps > 0 else 0.0
    checkpoint_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))

    print("\n===== Inference Summary =====")
    print(f"Checkpoint : {model_args.model_name_or_path} ({checkpoint_name})")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"Total      : {total} ({total_new} newly evaluated, {len(mc_df) - total_new} cached)")
    print(f"Correct    : {correct_answers}/{mc_steps} ({accuracy:.2%}) [MC only, newly evaluated]")
    print(f"Wrong fmt  : {wrong_format_answers}")
    print(f"\nPer-category accuracy (MC only):")
    for cat in sorted(cat_total):
        n = cat_total[cat]
        c = cat_correct.get(cat, 0)
        print(f"  {cat:<25} {c}/{n} ({c/n:.2%})")
    print(f"\nOutput     : {args.output_path}")


if __name__ == "__main__":
    main()
