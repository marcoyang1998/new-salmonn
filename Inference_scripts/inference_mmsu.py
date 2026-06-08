import json
import math
import re
import warnings
import argparse
import os

from tqdm import tqdm
from datasets import load_dataset

from modeling_salmonn import SALMONN
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
import torch.nn.functional as F
from transformers import AutoFeatureExtractor
from inference_utils import (
    ModelArguments,
    OVERRIDE_KEYS,
    add_mc_prompt_style_arg,
    get_mc_prompt_instruction,
    maybe_init_qwen3_embedding_model,
    override_args_from_config,
)


DATASET_ROOT = "/mnt/shared-storage-user/brainllm-share/data/MMSU"
ZIPFORMER_LIKE = {"zipformer2", "spear_transformer"}

QUESTION_PROMPTS = (
    "Choose the most suitable answer from options A, B, C, and D. "
    "You must respond with only A, B, C, or D."
)

QUESTION_TEMPLATE = (
    # "Answer the following multiple-choice question using only the correct option.\n"
    "Listen to the audio and answer the following multiple-choice question.\n"
    "Question: {question}\n"
    "Choices:\n"
    "{choices_str}\n"
    "{instruction}"
)
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
    parser = argparse.ArgumentParser(description="Evaluate SALMONN on MMSU")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--encoder_type", type=str, default=None,
                        help="Audio encoder type. If not set, value from checkpoint config.json is used.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split (default: train)")
    parser.add_argument("--output_jsonl", type=str, required=True,
                        help="Path to save output JSONL file")
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument("--use_qa_prompt", type=str2bool, default=False,
                        help="Use the QA-style prompt template instead of the default MMSU prompt")
    add_mc_prompt_style_arg(parser)
    return parser.parse_args()


def build_prompt(question, choice_a, choice_b, choice_c, choice_d, use_qa_prompt=False, mc_prompt_style="neutral"):
    if use_qa_prompt:
        choices_str = (
            f"Option A: {choice_a}\n"
            f"Option B: {choice_b}\n"
            f"Option C: {choice_c}\n"
            f"Option D: {choice_d}"
        )
        return QUESTION_TEMPLATE.format(
            question=question,
            choices_str=choices_str,
            instruction=get_mc_prompt_instruction(mc_prompt_style),
        )
    choices = f"A. {choice_a}\nB. {choice_b}\nC. {choice_c}\nD. {choice_d}"
    return f"{QUESTION_PROMPTS}\n\nQuestion: {question}\n\n{choices}"


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


def extract_features_from_array(audio_array, sampling_rate, fbank, model_args):
    """Extract fbank features from a raw audio numpy array."""
    audio_chunk = model_args.audio_chunk * 16000
    enc = model_args.encoder_type

    feature, raw_wavs, audio_nums, split_feature_lens = [], [], [], []

    audio = torch.from_numpy(audio_array).float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # [1, T]

    if sampling_rate != 16000:
        audio = torchaudio.functional.resample(audio, sampling_rate, 16000)
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
        sf_audio = audio.squeeze().numpy()
        feature.append(fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
    elif enc in ("qwenomni", "qwen3omni"):
        sf_audio = audio.squeeze().numpy()
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


def extract_predicted_letter(model_output_raw, num_choices=4):
    """Parse a letter from model output. Returns (letter, is_wrong_format)."""
    single_letter_pattern = re.compile(r"^[A-Za-z]$")
    option_letter_pattern = re.compile(r"^Option\s+([A-Za-z])$", re.IGNORECASE)

    if single_letter_pattern.fullmatch(model_output_raw):
        return model_output_raw.upper(), False
    m = option_letter_pattern.fullmatch(model_output_raw)
    if m:
        return m.group(1).upper(), False

    # Fall back to first valid letter found
    first = re.search(r"[A-Za-z]", model_output_raw)
    if first:
        candidate = first.group(0).upper()
        letter = candidate if ord(candidate) - ord("A") < num_choices else "A"
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
    enc = model_args.encoder_type

    dataset = load_dataset(DATASET_ROOT, split=args.split)

    total_questions = len(dataset)
    correct_answers = 0
    wrong_format_answers = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)

    with open(args.output_jsonl, "w") as fout:
        for i, item in enumerate(tqdm(dataset)):
            audio = item["audio"]
            audio_array = audio["array"]
            sampling_rate = audio["sampling_rate"]
            audio_path = audio["path"]

            question = item["question"]
            choice_a = item["choice_a"]
            choice_b = item["choice_b"]
            choice_c = item.get("choice_c", "")
            choice_d = item.get("choice_d", "")

            prompt = build_prompt(
                question,
                choice_a,
                choice_b,
                choice_c,
                choice_d,
                use_qa_prompt=args.use_qa_prompt,
                mc_prompt_style=args.mc_prompt_style,
            )

            messages = [{"role": "user", "content": "<audio>" + prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            feature, raw_wavs, audio_nums, split_feature_lens = extract_features_from_array(
                audio_array, sampling_rate, fbank, model_args
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
            num_choices = sum(1 for c in [choice_a, choice_b, choice_c, choice_d] if c)
            predicted_letter, is_wrong_format = extract_predicted_letter(model_output_raw, num_choices)

            if is_wrong_format:
                wrong_format_answers += 1
                warnings.warn(
                    f"Invalid output format for {audio_path}. Got: {repr(model_output_raw)}",
                    stacklevel=1,
                )

            answer_gt = item["answer_gt"]
            # choices_list = [choice_a, choice_b, choice_c, choice_d]
            # gt_letter = next(
            #     (chr(ord("A") + i) for i, c in enumerate(choices_list) if c.strip() == answer_gt.strip()),
            #     None,
            # )
            # if gt_letter is not None and predicted_letter == gt_letter:
            #     correct_answers += 1

            result = {
                "id": item["id"],
                "audio_path": audio_path,
                "question": question,
                "choice_a": choice_a,
                "choice_b": choice_b,
                "choice_c": choice_c,
                "choice_d": choice_d,
                "answer_gt": answer_gt,
                "response": predicted_letter,
                "task_name": item["task_name"],
                "category": item["category"],
                "sub-category": item["sub-category"],
                "sub-sub-category": item["sub-sub-category"],
                "linguistics_sub_discipline": item["linguistics_sub_discipline"],
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            # if i % 100 == 0 and i:
            #     print(
            #         f"[{i}/{total_questions}] accuracy: {correct_answers}/{i} "
            #         f"({correct_answers/i:.2%}), wrong format: {wrong_format_answers}"
            #     )

    accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
    checkpoint_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))

    print("\n===== Inference Summary =====")
    print(f"Checkpoint : {model_args.model_name_or_path} ({checkpoint_name})")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"Split      : {args.split}")
    print(f"Total      : {total_questions}")
    print(f"Correct    : {correct_answers} ({accuracy:.2%})")
    print(f"Wrong fmt  : {wrong_format_answers}")
    print(f"Output     : {args.output_jsonl}")


if __name__ == "__main__":
    main()
