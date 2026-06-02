import os
import json
import argparse
from typing import List, Optional

from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ==========================================
# Prompt builder (inlined from generate_biasing_cot.py)
# ==========================================
def generate_cot_prompt(
    gt_biasing_list: List[str],
    biasing_list: List[str],
    transcript: Optional[str] = None,
) -> str:
    """Build the user prompt for CoT generation.

    Branches on whether any target words were heard:
    - Non-empty gt: includes the transcript for context, emphasises naming targets.
    - Empty gt: does NOT include the transcript (to prevent leakage) and uses a
      dedicated prompt that stresses the no-match conclusion only.
    """
    if gt_biasing_list:
        return _prompt_with_targets(gt_biasing_list, biasing_list, transcript)
    else:
        return _prompt_no_targets(biasing_list)


def _prompt_with_targets(
    gt_biasing_list: List[str],
    biasing_list: List[str],
    transcript: Optional[str],
) -> str:
    gt_words_str     = ", ".join(gt_biasing_list)
    biasing_list_str = ", ".join(biasing_list)

    if transcript:
        input_section = f'Transcript: "{transcript}"\nHeard Target Words: {gt_words_str}'
    else:
        input_section = f"Heard Target Words (from audio): {gt_words_str}"

    return f"""# System Role
You are an expert internal reasoning module for a state-of-the-art Speech-Language Model. Your task is to generate diverse Chain-of-Thought (CoT) reasoning paths that simulate how an ASR model processes a "Biasing List" while listening to speech.

# Task Description
I will provide you with:
1. Heard Target Words (the actual biasing words present in the audio).
2. Biasing List (a list of reference words, containing both targets and distractors).

You need to generate 4 DIFFERENT kinds of reasoning paths enclosed in specific tags.

# Rules for the CoT:
1. Positive Matches: You MUST explicitly mention the words from the "Heard Target Words" list, simulating that you clearly detected their acoustic features.
2. Distractor Handling: DO NOT list every single distractor. You can casually mention 1 or 2 distractors as being absent, and then use a generalized statement to dismiss the rest (e.g., "The other reference words do not match the acoustic signal").
3. Output ONLY the 4 reasoning blocks.
4. Mandatory Transition: You MUST ALWAYS end every reasoning path by explicitly stating that you are now ready to transcribe the speech. NEVER say "no further action is needed."
5. No Transcript Spoilers: You should only mention the specific matched target words. DO NOT quote or summarize the rest of the spoken transcript.

# Required Variations for the 4 Paths:
- <cot_path_A> (Confident & Direct): Quickly identify the targets and dismiss the rest in a straightforward manner.
- <cot_path_B> (Acoustic/Phonetic Simulation): Simulate checking the pronunciation or acoustic boundaries of the target words against the audio, then dismissing the distractors.
- <cot_path_C> (Scanning & Elimination): Simulate scanning the list left-to-right, picking out the targets, and concluding the rest are pure distractors.
- <cot_path_D> (Contextual Verification): Acknowledge the targets fit the auditory context, while explicitly stating the bulk of the distractors are irrelevant.

# Example Input:
Heard Target Words (from audio): meme, humbugging, pinocchio's
Biasing List: meme, humbugging, pinocchio's, more's, ayme, voxels, tweed, combats, minimums, charms, poster, lowry, rish, roused, sandstones, oasis, moneyline, lordship's, baltic, cnu

# Example Output:
<cot_path_A>
I clearly hear "humbugging", "pinocchio's", and "meme" in the audio stream. These match the reference list perfectly. As for words like "voxels" or "lowry", I don't hear them at all. The rest of the distractors in the list are definitely not present in this speech segment. Proceeding to transcribe.
</cot_path_A>

<cot_path_B>
Checking the acoustic signals against the provided list. The phonetic structures for "humbugging" and "pinocchio's" align perfectly with the middle segment of the audio. The utterance ends with a clear /m/ sound matching "meme". The other items in the list, such as "tweed" and the remaining distractors, have no acoustic evidence here. I will ignore them. Ready to transcribe.
</cot_path_B>

<cot_path_C>
Scanning the biasing list: "meme" is a hit. "humbugging" is a hit. "pinocchio's" is also a hit. I don't detect "more's" or "ayme". In fact, looking at the rest of the 15+ words in the list, none of them match the audio input. They are just distractors. I'll focus on the three confirmed words for the transcription.
</cot_path_C>

<cot_path_D>
The speaker is talking about something specific. I catch the words "humbugging" and "pinocchio's", followed by "meme" at the end. These fit the acoustic context seamlessly. The provided list contains many other words, but after a quick cross-check, the remaining distractors are completely absent from this recording. Ready to output the text.
</cot_path_D>

---
# Your Turn
{input_section}
Biasing List: {biasing_list_str}
"""


def _prompt_no_targets(biasing_list: List[str]) -> str:
    biasing_list_str = ", ".join(biasing_list)

    return f"""# System Role
You are an expert internal reasoning module for a state-of-the-art Speech-Language Model. Your task is to generate diverse Chain-of-Thought (CoT) reasoning paths that simulate how an ASR model processes a "Biasing List" while listening to speech — specifically for the case where NONE of the biasing words were heard.

# Task Description
I will provide you with:
1. Biasing List (a list of reference words, none of which appear in the audio).

You need to generate 4 DIFFERENT kinds of reasoning paths enclosed in specific tags.

# Rules for the CoT:
1. No Matches: You MUST conclude that NONE of the words in the biasing list were detected. Do NOT invent false positives.
2. CRITICAL — No Transcript Leakage: You MUST NOT quote, paraphrase, or reproduce ANY part of the actual spoken content. Do NOT describe or summarize what the speaker said. Reason only from the acoustic signal and the biasing list — never from the transcript text.
3. Distractor Handling: You may mention 1-2 specific words from the biasing list as examples of what was NOT heard, then dismiss the rest generally.
4. Mandatory Transition: You MUST ALWAYS end every reasoning path by explicitly stating you are ready to transcribe. NEVER say "no further action is needed."

# Required Variations for the 4 Paths:
- <cot_path_A> (Confident & Direct): Quickly conclude no targets were found and dismiss the entire list.
- <cot_path_B> (Acoustic/Phonetic Simulation): Simulate checking the acoustic signal against the list and finding no phonetic matches.
- <cot_path_C> (Scanning & Elimination): Simulate scanning the list left-to-right and concluding every entry is a distractor.
- <cot_path_D> (Contextual Verification): Acknowledge the audio context does not align with any of the biasing words, without describing what the audio content actually is.

# Example Input:
Heard Target Words: None
Biasing List: meme, humbugging, pinocchio's, more's, ayme, voxels, tweed, combats, minimums, charms, poster, lowry, rish, roused, sandstones, oasis, moneyline, lordship's, baltic, cnu

# Example Output:
<cot_path_A>
I listened carefully to the entire audio segment. None of the words in the biasing list — such as "meme", "humbugging", or "voxels" — were present in the speech. The entire list consists of distractors for this recording. Proceeding to transcribe.
</cot_path_A>

<cot_path_B>
Checking the acoustic signal against each entry in the biasing list. No phonetic structure in the audio matches "meme", "humbugging", or "pinocchio's". Scanning the remaining entries yields the same result — no acoustic evidence for any listed word. I am ready to transcribe.
</cot_path_B>

<cot_path_C>
Scanning the biasing list from left to right: "meme" — not detected. "humbugging" — not detected. "pinocchio's" — not detected. Continuing through the full list, none of the entries produce a match. All words are distractors for this audio segment. Ready to transcribe.
</cot_path_C>

<cot_path_D>
The auditory context of this recording does not align with any word in the biasing list. I checked entries such as "meme" and "humbugging" — neither fits the acoustic signal here. The remaining entries are equally absent. Proceeding to transcribe.
</cot_path_D>

---
# Your Turn
Heard Target Words: None
Biasing List: {biasing_list_str}
"""

# ==========================================
# Args
# ==========================================
_SCRIPT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser(description="Generate CoT think-paths for biasing word lists using Qwen3-8B.")
parser.add_argument("--input",          default=os.path.join(_SCRIPT_DIR, "data", "test_with_biasing.json"),     help="Path to input JSON file")
parser.add_argument("--output",         default=os.path.join(_SCRIPT_DIR, "data", "test_with_biasing_cot.json"), help="Path to output JSON file")
parser.add_argument("--model",          default="Qwen/Qwen3-8B",  help="HuggingFace model name or local path")
parser.add_argument("--batch-size",     type=int, default=4,       help="Number of items to process per GPU batch")
parser.add_argument("--max-new-tokens", type=int, default=2048,    help="Maximum new tokens to generate per item")
parser.add_argument("--thinking",       action="store_true",       help="Enable Qwen3 internal <think> chain (thinking mode)")
args = parser.parse_args()

ENABLE_THINKING = args.thinking
BATCH_SIZE      = args.batch_size
INPUT_JSON      = args.input
OUTPUT_JSON     = args.output
MODEL_NAME      = args.model
MAX_NEW_TOKENS  = args.max_new_tokens

# ==========================================
# Load model
# ==========================================
print(f"Loading tokenizer from {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Left-pad so all sequences end at the same position — required for batched causal LM decoding
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

print(f"Loading model from {MODEL_NAME} ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map=device,
)
model.eval()
print(f"Thinking mode: {'ON' if ENABLE_THINKING else 'OFF'}, Batch size: {BATCH_SIZE}\n")

# ==========================================
# Helper: parse think paths into a list
# ==========================================
import re
_PATH_TAGS = ["cot_path_A", "cot_path_B", "cot_path_C", "cot_path_D"]

def parse_think_paths(response: str) -> List[str]:
    """Extract the 4 <cot_path_X>...</cot_path_X> blocks into an ordered list.
    The last tag often lacks a closing tag (model emits EOS instead), so we
    fall back to matching from the opening tag to end-of-string for that case.
    """
    paths = []
    for i, tag in enumerate(_PATH_TAGS):
        if i < len(_PATH_TAGS) - 1:
            match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        else:
            # Last path: accept with or without closing tag
            match = re.search(rf"<{tag}>(.*?)(?:</{tag}>|$)", response, re.DOTALL)
        paths.append(match.group(1).strip() if match else "")
    return paths

# ==========================================
# Helper: batched generation
# ==========================================
def generate_batch(prompts: List[str]) -> List[str]:
    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        for p in prompts
    ]
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    input_lengths = model_inputs.input_ids.shape[1]
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    # Strip input tokens — with left-padding all sequences share the same padded
    # input length, so slicing at input_lengths works uniformly across the batch.
    responses = []
    for seq in generated_ids:
        new_ids = seq[input_lengths:]
        # Decode without stripping special tokens, then manually remove chat tokens.
        # skip_special_tokens=True can silently strip parts of XML-like tags that
        # share a prefix with Qwen3 special tokens (e.g. </think> inside </think_path_D>).
        raw = tokenizer.decode(new_ids, skip_special_tokens=False)
        # Strip Qwen3 chat/control tokens that may appear at the tail
        for tok in ["<|im_end|>", "<|endoftext|>", "</think>", "<think>"]:
            raw = raw.replace(tok, "")
        responses.append(raw.strip())
    return responses

# ==========================================
# Iterate over dataset
# ==========================================
print(f"Reading {INPUT_JSON} ...")
with open(INPUT_JSON, "r") as f:
    dataset = json.load(f)

items = dataset["data"]

def build_prompt(item: dict) -> str:
    gt_list   = item.get("ground_truth_biasing_list", [])
    bias_list = item.get("biasing_list", [])
    transcript = None
    for msg in item.get("messages", []):
        if msg["role"] == "assistant":
            transcript = msg["content"]
            break
    return generate_cot_prompt(
        gt_biasing_list=gt_list,
        biasing_list=bias_list,
        transcript=transcript,
    )

# Process in batches
with tqdm(total=len(items), desc="Generating CoT", unit="item") as pbar:
    for batch_start in range(0, len(items), BATCH_SIZE):
        batch = items[batch_start : batch_start + BATCH_SIZE]
        prompts = [build_prompt(item) for item in batch]
        responses = generate_batch(prompts)
        for item, response in zip(batch, responses):
            item["think_paths"] = parse_think_paths(response)
        pbar.update(len(batch))

# ==========================================
# Save results
# ==========================================
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved results to {OUTPUT_JSON}")
