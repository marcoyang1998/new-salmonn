"""
Convert MMAU test JSON to SALMONN MC_QA format.

Source format (mmau-test.json):
  {
    "id": ...,
    "audio_id": "./test-audios/<id>.wav",
    "question": "...",
    "answer": "",          # empty for test split
    "choices": ["opt1", "opt2", ...],
    "dataset": ...,
    "task": ...,
    "category": ...,
    ...
  }

Target format (GeminiQA_MC_val_one_letter.json):
  {
    "annotation": [
      {
        "path": "/abs/path/to/audio.wav",
        "text": "A",        # answer letter; empty string if unknown
        "task": "MC_QA",
        "Q": "Answer the following multiple-choice question ...\nQuestion: ...\nChoices:\nOption A: ...\n...\nPlease output your final answer ...",
        # extra MMAU fields preserved
      },
      ...
    ]
  }
"""

import json
import os
import argparse


QUESTION_TEMPLATE = (
    "Answer the following multiple-choice question using only the correct option.\n"
    "Question: {question}\n"
    "Choices:\n"
    "{choices_str}\n"
    "Please output your final answer with a single letter. "
    "For example, if you think the answer is Option A, please just output 'A'"
)

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def build_q_string(question: str, choices: list[str]) -> str:
    choices_lines = "\n".join(
        f"Option {LETTERS[i]}: {choice}" for i, choice in enumerate(choices)
    )
    return QUESTION_TEMPLATE.format(question=question, choices_str=choices_lines)


def answer_to_letter(answer: str, choices: list[str]) -> str:
    """Map a choice string or letter to the corresponding letter label."""
    if not answer:
        return ""
    # If it's already a single letter that matches one of the options
    if len(answer) == 1 and answer.upper() in LETTERS[: len(choices)]:
        return answer.upper()
    # Otherwise match by string content
    for i, choice in enumerate(choices):
        if choice.strip().lower() == answer.strip().lower():
            return LETTERS[i]
    return ""


def convert(
    input_path: str,
    output_path: str,
    audio_root: str | None = None,
) -> None:
    with open(input_path) as f:
        mmau_data = json.load(f)

    annotations = []
    for item in mmau_data:
        choices = item.get("choices", [])
        question = item.get("question", "")
        answer = item.get("answer", "")

        # Resolve audio path
        audio_id = item.get("audio_id", "")
        if audio_root:
            # Replace the leading "./" with the provided root
            rel = audio_id.lstrip("./")
            path = os.path.join(audio_root, rel)
        else:
            path = audio_id

        entry = {
            "path": path,
            "text": answer_to_letter(answer, choices),
            "task": "MC_QA",
            "Q": build_q_string(question, choices),
        }

        # Preserve all original MMAU fields as extra attributes
        for key, val in item.items():
            if key not in ("audio_id", "question", "answer", "choices"):
                entry[key] = val

        annotations.append(entry)

    output = {"annotation": annotations}
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(annotations)} items → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MMAU JSON to SALMONN format")
    parser.add_argument("input", help="Path to mmau-test.json")
    parser.add_argument("output", help="Path for the converted output JSON")
    parser.add_argument(
        "--audio-root",
        default=None,
        help=(
            "Absolute root directory for audio files. "
            "If provided, the relative paths in audio_id are resolved against this root. "
            "E.g. /mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/MMAU"
        ),
    )
    args = parser.parse_args()
    convert(args.input, args.output, args.audio_root)
