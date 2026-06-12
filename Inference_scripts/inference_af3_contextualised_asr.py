import json
import re
from argparse import ArgumentParser
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor


DEFAULT_MODEL = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/models/nvidia--audio-flamingo-3-hf"
DEFAULT_TEST_JSON = "salmonn_data_v1.1/contextual_biasing/ASR_gigaspeech_train_first1000_contextualised_asr_biasing_le10_top20.json"
DEFAULT_RESULTS_FOLDER = "results_contextualised_asr"


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError("Boolean value expected.")


def parse_args():
    parser = ArgumentParser(description="Run Audio Flamingo 3 contextualised ASR inference.")
    parser.add_argument("--test_set_path", type=str, default=DEFAULT_TEST_JSON)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--model_id", type=str, default="audio_flamingo3_train_first1000")
    parser.add_argument("--results_folder", type=str, default=DEFAULT_RESULTS_FOLDER)
    parser.add_argument("--write_path_title", type=str, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use_biasing_list", type=str2bool, default=True)
    return parser.parse_args()


def build_prompt(sample, use_biasing_list=True):
    if not use_biasing_list:
        return "Transcribe the input speech."

    biasing_list = sample.get("biasing_list", [])
    bias_words_text = f"[{', '.join(str(word) for word in biasing_list)}]"
    biasing_text = "Pay extra attention to the following contextual words:\n" + bias_words_text
    return f"Transcribe the input speech. {biasing_text}"


def build_conversation(sample, use_biasing_list=True):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt(sample, use_biasing_list)},
                {"type": "audio", "path": sample["path"]},
            ],
        }
    ]


def generate_response(processor, model, sample, max_new_tokens, use_biasing_list=True):
    inputs = processor.apply_chat_template(
        build_conversation(sample, use_biasing_list),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)

    decoded_outputs = processor.batch_decode(
        outputs[:, inputs.input_ids.shape[1]:],
        skip_special_tokens=True,
    )
    return decoded_outputs[0].strip() if decoded_outputs else ""


def clean_af3_response(response):
    response = response.strip()
    patterns = [
        r"^the spoken content of the audio is\s+(['\"])(.*)\1\.?$",
        r"^the transcription of the audio is\s+(['\"])(.*)\1\.?$",
        r"^the content of the input audio is\s+(['\"])(.*)\1\.?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, response, flags=re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(2).strip()
    return response


def load_af3_model(model_path):
    from transformers import AudioFlamingo3ForConditionalGeneration

    return AudioFlamingo3ForConditionalGeneration.from_pretrained(model_path, device_map="auto")


def main():
    args = parse_args()
    write_path_title = args.write_path_title or f"stage2_{args.model_id}_contextualised_asr"
    write_path = Path(args.results_folder) / f"{write_path_title}.jsonl"
    write_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor from {args.model} ...")
    processor = AutoProcessor.from_pretrained(args.model)
    print(f"Loading model from {args.model} ...")
    model = load_af3_model(args.model)
    model.eval()

    with open(args.test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)["annotation"]

    data = data[args.start:]
    if args.limit is not None:
        data = data[:args.limit]

    with open(write_path, "a", encoding="utf-8") as out_file:
        for sample in tqdm(data, desc="Inferencing Audio Flamingo 3"):
            result = dict(sample)
            response = generate_response(
                processor,
                model,
                sample,
                args.max_new_tokens,
                args.use_biasing_list,
            )
            result["response"] = clean_af3_response(response)
            json.dump(result, out_file, ensure_ascii=False)
            out_file.write("\n")
            out_file.flush()

    print(f"Saved results to {write_path}")


if __name__ == "__main__":
    main()