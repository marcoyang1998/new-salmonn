import argparse
import json
import logging
import os
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from inference_utils import (
    ModelArguments,
    extract_audio_features,
    get_fbank,
    maybe_init_qwen3_embedding_model,
    override_args_from_config,
)
from modeling_salmonn import SALMONN


logging.basicConfig(level=logging.ERROR, force=True)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run SALMONN contextual-biasing ASR inference.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to contextual-biasing JSON, e.g. ASR_gigaspeech_contextual_ASR_biasing_le12_top100.json.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to write JSON with model_prediction added to each processed item.",
    )
    parser.add_argument(
        "--encoder_type",
        type=str,
        default=None,
        help="Audio encoder type override. If unset, checkpoint config/defaults are used.",
    )
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument("--split_audio", type=str2bool, default=False)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--disable_thinking", type=str2bool, default=True)
    parser.add_argument("--use_ctx_audio", type=str2bool, default=False)
    parser.add_argument("--save_every", type=int, default=1000, help="Write partial results every N processed samples.")

    parser.add_argument("--use_beam_search", type=str2bool, default=False)
    parser.add_argument("--beam_size", type=int, default=4)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--use_nucleus_sampling", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def get_data_items(payload: Any):
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        if "data" in payload:
            return payload["data"], "data"
        if "annotation" in payload:
            return payload["annotation"], "annotation"
    raise ValueError("Input JSON must be a list or a dict containing a 'data' or 'annotation' list.")


def get_base_instruction(sample: dict) -> str:
    messages = sample.get("messages", [])
    if messages and isinstance(messages[0], dict):
        content = str(messages[0].get("content", "")).replace("<audio>", "").strip()
        if content:
            return content
    return "Recognize the speech and give me the transcription."


def format_biasing_list(biasing_list) -> str:
    if isinstance(biasing_list, list):
        return f"[{', '.join(str(word) for word in biasing_list)}]"
    if biasing_list is None:
        return "[]"
    return f"[{biasing_list}]"


def build_prompt(sample: dict, use_ctx_audio: bool = True) -> str:
    biasing_list = sample.get("biasing_list", [])
    ctx_audios = sample.get("ctx_audios", [])
    base_instruction = get_base_instruction(sample).rstrip()
    if base_instruction and base_instruction[-1] not in ".!?:":
        base_instruction += "."

    if use_ctx_audio and ctx_audios:
        if not isinstance(biasing_list, list) or not isinstance(ctx_audios, list):
            raise ValueError("Context-audio biasing requires list fields: biasing_list and ctx_audios.")
        if len(biasing_list) != len(ctx_audios):
            raise ValueError(
                "Context-audio biasing requires one ctx_audio per biasing word, "
                f"but got {len(ctx_audios)} ctx_audios and {len(biasing_list)} biasing words."
            )
        lines = [
            base_instruction,
            "Use the following contextual words and their pronunciations as references while transcribing the speech:",
            "<biasing_list>",
        ]
        for word in biasing_list:
            lines.append(f"<audio>{word}")
        lines.append("</biasing_list>.")
        return "\n".join(lines)

    return "\n".join(
        [
            base_instruction,
            "Pay extra attention to the following contextual words:",
            "<biasing_list>",
            format_biasing_list(biasing_list),
            "</biasing_list>.",
        ]
    )


def get_main_audio_paths(sample: dict) -> list[str]:
    if "audios" in sample:
        return list(sample["audios"])
    if "path" in sample:
        audio_paths = [sample["path"]]
        audio_paths.extend(sample.get("expand_wav", []))
        return audio_paths
    raise KeyError("Each sample must contain either 'audios' or 'path'.")


def get_audio_paths(sample: dict, use_ctx_audio: bool = True) -> list[str]:
    audio_paths = get_main_audio_paths(sample)
    if use_ctx_audio and sample.get("ctx_audios"):
        ctx_audios = sample["ctx_audios"]
        if not isinstance(ctx_audios, list):
            raise ValueError("ctx_audios must be a list when present.")
        audio_paths.extend(ctx_audios)
    return audio_paths


def prepare_model_inputs(texts: list[str], audio_nums: list[int], tokenizer, model, model_args):
    if model_args.llm_type == "Llama":
        processed_texts = [
            text.replace("<audio>", "<|reserved_special_token_0|><|reserved_special_token_1|>")
            for text in texts
        ]
    elif audio_nums:
        processed_texts = []
        text_index = 0
        text = texts[text_index]
        for audio_num in audio_nums:
            while "<audio>" not in text:
                processed_texts.append(text)
                text_index += 1
                text = texts[text_index]
            text = text.replace("<audio>", "<|vision_start|>" * audio_num + "<|vision_end|>", 1)
        processed_texts.append(text)
        processed_texts.extend(texts[len(processed_texts):])
    else:
        processed_texts = [text.replace("<audio>", "<|vision_start|><|vision_end|>") for text in texts]

    return tokenizer(processed_texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)


def clean_model_output(generated_id, tokenizer) -> str:
    output_ids = generated_id.tolist()
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
        output_ids = output_ids[index:]
    except ValueError:
        pass
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    content = content.split("</think>")[-1].strip()
    if content.startswith("<think>"):
        content = content.replace("<think>", "", 1).strip()
    return content


def get_generation_config(args):
    if args.use_beam_search:
        return {
            "num_beams": args.beam_size,
            "length_penalty": args.length_penalty,
            "early_stopping": True,
            "do_sample": False,
        }
    if args.use_nucleus_sampling:
        return {
            "do_sample": True,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
        }
    return {}


def load_model_and_tokenizer(args):
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
        raise ValueError(f"Unknown llm_type: {model_args.llm_type}")

    model = SALMONN.from_pretrained(
        model_args.model_name_or_path,
        config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path, "config.json")),
        model_args=model_args,
        torch_dtype="auto",
        device_map="auto",
    )
    if getattr(model_args, "inject_temporal_embedding", False):
        model.register_temporal_tokens(tokenizer)
    if getattr(model_args, "inject_temporal_embedding_nl", False):
        model.register_nl_timestamp_tokenizer(tokenizer)
    maybe_init_qwen3_embedding_model(model, model_args)
    model.eval()
    return model, tokenizer, model_args


def write_payload(payload, output_path: str):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(payload, out_file, indent=2, ensure_ascii=False)


def generate_batch(batch, model, tokenizer, fbank, model_args, args):
    texts = []
    user_prompts = []
    audio_paths = []
    for sample in batch:
        prompt = build_prompt(sample, use_ctx_audio=args.use_ctx_audio)
        user_prompts.append(prompt)
        main_audio_num = len(get_main_audio_paths(sample))
        audio_paths.extend(get_audio_paths(sample, use_ctx_audio=args.use_ctx_audio))
        messages = [{"role": "user", "content": "<audio>" * main_audio_num + prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=not args.disable_thinking,
        )
        texts.append(text)

    features, feature_lens, raw_wavs, audio_nums = extract_audio_features(
        audio_paths,
        fbank,
        model,
        model_args,
        split_audio=args.split_audio,
    )
    model_inputs = prepare_model_inputs(texts, audio_nums, tokenizer, model, model_args)
    generation_config = get_generation_config(args)

    with torch.no_grad():
        torch.manual_seed(args.seed)
        generated_ids = model.generate(
            **model_inputs,
            fbank_feature=features,
            fbank_feature_len=feature_lens,
            raw_wavs=raw_wavs,
            user_prompts=user_prompts,
            max_new_tokens=args.max_new_tokens,
            **generation_config,
        )

    return [clean_model_output(generated_id, tokenizer) for generated_id in generated_ids]


def main():
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as in_file:
        payload = json.load(in_file)
    data, _ = get_data_items(payload)

    end = len(data) if args.limit is None else min(len(data), args.start + args.limit)
    indices = list(range(args.start, end))
    if not indices:
        write_payload(payload, args.output_path)
        print(f"No samples selected. Wrote unchanged JSON to {args.output_path}")
        return

    model, tokenizer, model_args = load_model_and_tokenizer(args)
    fbank = get_fbank(model_args)
    print(f"Encoder type: {model_args.encoder_type}")
    print(f"Samples     : {args.start}..{end - 1} ({len(indices)} total)")

    processed = 0
    for batch_start in tqdm(range(0, len(indices), args.batch_size), desc="Inferencing contextual biasing"):
        batch_indices = indices[batch_start:batch_start + args.batch_size]
        batch = [data[index] for index in batch_indices]
        predictions = generate_batch(batch, model, tokenizer, fbank, model_args, args)
        for index, prediction in zip(batch_indices, predictions):
            data[index]["model_prediction"] = prediction
        processed += len(batch_indices)
        if args.save_every > 0 and processed % args.save_every == 0:
            write_payload(payload, args.output_path)

    write_payload(payload, args.output_path)
    print("\n===== Inference Summary =====")
    print(f"Checkpoint : {model_args.model_name_or_path}")
    print(f"Input      : {args.input_json}")
    print(f"Output     : {args.output_path}")
    print(f"Processed  : {processed}")


if __name__ == "__main__":
    main()