import json
from tqdm import tqdm
import random

from lhotse import load_manifest_lazy

# asr_prompts = [
#     "Can you write down the transcription of the speech?",
#     "Write down the content of the speech you heard.",
#     "Can you transcribe the speech into a written format?",
#     "Give me the transcription of the speech you heard.",
#     "What is the content of the speech you heard?",
#     "Listen to the speech and recognize its content.",
#     "Put the speech into a written format.",
#     "Please help me to transcribe the speech into a written format.",
#     "Can you recognize what you heard in the speech?",
#     "Please transcribe the speech into a written format.",
#     "Please write down the transcription of the speech.",
#     "Listen to the speech and write down its content.",
#     "Recognize the content of the speech you heard.",
#     "Recognize the speech and give me the transcription.",
#     "Recognize the speech and write it down in a written format."
# ]

asr_prompts = [
    "Please write down the transcription of the speech.",
    "Please transcribe the speech into a written format.",
    "Write down the content of the speech you heard.",
    "Transcribe the speech.",
]

CV_ROOT = "/mnt/shared-storage-gpfs2/speechllm-share/data/common_voice_17_0"


def ends_with_proper_punctuation(text: str, allowed_punctuation=(".", "?", "!")) -> bool:
    text = text.rstrip()
    if not text:
        return False
    return text.endswith(allowed_punctuation)

def convert_commonvoice_english_to_salmonn_json(args):
    cuts = load_manifest_lazy(args.manifest_path)
    
    all_entries = []
    max_text_len = 0
    longest_text = ""
    
    for i, cut in tqdm(enumerate(cuts)):
        audio_path = cut.recording.sources[0].source
        audio_path = audio_path.replace(
            "download/common_voice_17_0",
            CV_ROOT,
        )
        text = cut.supervisions[0].text.rstrip()
        # if not ends_with_proper_punctuation(text):
        #     text = f"{text}." if text else "."
            
        if len(text) > max_text_len:
            max_text_len = max(max_text_len, len(text))
            longest_text = text
        
        prompt = random.sample(asr_prompts, 1)[0]
        prompt = "<audio>" + prompt # add placeholder for audio embeddings
        current_entry = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                },
                {
                    "role": "assistant",
                    "content": text # this is a placeholder, we will sample from all captions during training
                }
            ],
            "audios":[
                audio_path,
            ],
            "durations": [
                cut.duration
            ],
            "task_type": "asr",
        }
        all_entries.append(current_entry)
        
    with open(args.output_json_path, "w", encoding="utf-8") as out_file:
        json.dump({"data": all_entries}, out_file, ensure_ascii=True, indent=2)
        
    print(f"The longest text content length is {longest_text} with {max_text_len} characters.")
    print(f"Saved the converted dataset to {args.output_json_path}. Total entries: {len(all_entries)}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert common voice Lhotse manifest to SALMONN json format")
    parser.add_argument("--manifest_path", type=str, required=True, help="Path to the common voice Lhotse manifest (e.g., cuts.jsonl.gz)")
    parser.add_argument("--output_json_path", type=str, required=True, help="Path to save the converted json file")
    
    args = parser.parse_args()
    convert_commonvoice_english_to_salmonn_json(args)