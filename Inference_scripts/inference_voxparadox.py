"""SALMONN inference on VoxParadox; writes raw responses for official scoring."""
import argparse
import json
import os

from voxparadox_utils import DATASET_ROOT, iter_records


def parse_args():
    from inference_utils import add_mc_prompt_style_arg
    from inference_mmsu import str2bool
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--dataset", default=os.path.join(DATASET_ROOT, "metadata.jsonl"))
    parser.add_argument("--audio_root", default=DATASET_ROOT)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--encoder_type", default=None)
    parser.add_argument("--concat_encoder_features", type=str2bool, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional bounded smoke run")
    parser.add_argument(
        "--extra_instruction", default="",
        help="Instruction inserted after the question and before the choices",
    )
    add_mc_prompt_style_arg(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max_samples must be positive")
    import soundfile as sf
    import torch
    from transformers import AutoConfig, AutoTokenizer
    from modeling_salmonn import SALMONN
    from inference_utils import ModelArguments, override_args_from_config, maybe_init_qwen3_embedding_model
    from inference_mmsu import (ZIPFORMER_LIKE, get_fbank, extract_features_from_array,
                               prepare_model_inputs, parse_model_output)
    from inference_mmar import build_prompt
    from itertools import islice

    model_args = ModelArguments()
    model_args.model_name_or_path = args.model_name_or_path
    model_args = override_args_from_config(args.model_name_or_path, model_args)
    if args.lora_alpha is not None:
        model_args.lora_alpha = args.lora_alpha
        print(f"Overriding lora_alpha with value from command line: {model_args.lora_alpha}")

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
    model.register_backbone_tokenizer(tokenizer)
    if model_args.inject_temporal_embedding:
        model.register_temporal_tokens(tokenizer)
    if getattr(model_args, "inject_temporal_embedding_nl", False):
        model.register_nl_timestamp_tokenizer(tokenizer)
    maybe_init_qwen3_embedding_model(model, model_args)
    model.eval()

    fbank = get_fbank(model_args)
    enc = model_args.encoder_type

    records = iter_records(args.dataset)
    if args.max_samples is not None:
        records = islice(records, args.max_samples)
    count = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    with open(args.output_jsonl, "w", encoding="utf-8") as fout:
        for item in records:
            choices = [item[f"choice_{letter}"] for letter in "abcd"]
            question = item["question"]
            if args.extra_instruction.strip():
                question = f"{question}\n{args.extra_instruction.strip()}"
            prompt = build_prompt(question, choices, mc_prompt_style=args.mc_prompt_style)
            audio_path = os.path.join(args.audio_root, item.get("audio_path") or item["file_name"])
            # Decode only this clip; downmix before the shared resampling frontend.
            audio_array, sampling_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            audio_array = audio_array.mean(axis=1)
            messages = [{"role": "user", "content": "<audio>" + prompt}]
            from backbone_compat import render_chat
            text = render_chat(tokenizer, messages, add_generation_prompt=True, enable_thinking=False)

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

            model_output_raw = parse_model_output(generated_ids, tokenizer, model)

            result = dict(item)
            result["audio_path"] = audio_path
            result["response"] = model_output_raw
            result["extra_instruction"] = args.extra_instruction
            result["prompt"] = prompt
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} examples", flush=True)
    print(f"Wrote {count} predictions to {args.output_jsonl}")


if __name__ == "__main__":
    main()
