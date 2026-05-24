import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Inference_scripts.inference_beans_dogs import (
    evaluate,
    extract_prediction,
    get_feature_tensors,
    get_label_list,
    load_model,
    normalize_label,
    select_split,
    summarize_metrics,
)
from Inference_scripts.inference_mmau import get_fbank, str2bool


PROMPT_TEMPLATE = "Describe the emotion of the speaker in one word."

def parse_args():
    parser = argparse.ArgumentParser(description="Run SALMONN emotion classification on IEMOCAP.")
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--iemocap_json",
        type=str,
        default="data/iemocap_emotion_test_salmonn.json",
        help="Path to the IEMOCAP SALMONN-format JSON.",
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
    parser.add_argument(
        "--prompt",
        type=str,
        default=PROMPT_TEMPLATE,
        help="Prompt text to use for IEMOCAP inference.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=32, help="Maximum number of generated tokens.")
    return parser.parse_args()


def build_prompt(prompt):
    return prompt


def main():
    args = parse_args()

    model, tokenizer, model_args = load_model(
        model_name_or_path=args.model_name_or_path,
        encoder_type=args.encoder_type,
        concat_encoder_features=args.concat_encoder_features,
    )
    fbank = get_fbank(model_args)

    with open(args.iemocap_json, "r") as handle:
        payload = json.load(handle)

    all_data = payload["data"] if isinstance(payload, dict) else payload
    data = select_split(all_data, args.split)
    if args.limit is not None:
        data = data[: args.limit]
    if not data:
        raise ValueError(f"No samples found in {args.iemocap_json} for split={args.split}")

    labels = get_label_list(payload if isinstance(payload, dict) else {"data": all_data})
    prompt = build_prompt(args.prompt)
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

    output_payload: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {"data": annotated_data}
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
    with open(args.output_path, "w") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=False)

    print("\n===== IEMOCAP Inference Summary =====")
    print(f"Checkpoint : {args.model_name_or_path}")
    print(f"Encoder    : {model_args.encoder_type}")
    print(f"Split      : {args.split}")
    print(f"Total      : {summary['num_samples']}")
    print(f"Parsed     : {summary['num_parsed_predictions']} ({summary['parsed_rate']:.2%})")
    print(f"Correct    : {summary['num_correct']} ({summary['accuracy']:.2%})")
    print(f"Output     : {args.output_path}")


if __name__ == "__main__":
    main()