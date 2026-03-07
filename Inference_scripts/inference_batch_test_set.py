import os
import json
from tqdm import tqdm
import logging
from inference_manager import InferenceManager
from argparse import ArgumentParser
from pathlib import Path

logging.basicConfig(level=logging.ERROR, force=True)

def main(test_set_path, write_path, batch_size, checkpoint_path, max_new_tokens: int = 500, start: int = 0, task_filter: list[str] = None, split_audio: bool = False):
    inference_manager = InferenceManager(checkpoint_path=checkpoint_path, max_new_tokens=max_new_tokens, task_filter=task_filter, split_audio=split_audio)

    with open(test_set_path, "r") as f:
        data = json.load(f)["annotation"]
    
    data = data[start:]
    with open(write_path, "a", encoding="utf-8") as out_file:
        batch_data = []
        for i, sample in enumerate(tqdm(data, desc="Inferencing Test Set")):
            batch_data.append(sample)
            if len(batch_data) == batch_size:
                inference_manager.process_batch(batch_data, out_file)
                batch_data = []
                print(f"processed {i + 1} samples")
                out_file.flush()

        if batch_data:
            inference_manager.process_batch(batch_data, out_file)
            print(f"processed {len(data)} samples")

if __name__ == "__main__":
    # all_tasks = [
    #                 'keywords',
    #                 'audiocaption', 
    #                 'audiocaption_v2', 
    #                 'asr', 
    #                 'translation_ec', 
    #                 'phone_recognition', 
    #                 'speech_query', 
    #                 'speaker_verification', 
    #                 'emotion_recognition', 
    #                 'gender_QA', 
    #                 'speech_separation', 
    #                 'slot_filling', 
    #                 'music_description'
    #                 ]
    default_tasks = [
                     'audiocaption', 
                     'audiocaption_v2', 
                     'asr', 
                     'translation_ec', 
                     'phone_recognition', 
                     'speaker_verification', 
                     'emotion_recognition', 
                     'speech_separation',
                     'music_description'
                     ]
    parser = ArgumentParser()
    # with default settings
    parser.add_argument("--test_set_path", type=str, default = "/mnt/bn/audio-visual-llm-data/datasets/multitask_json/test_AsrAacAstQueryPhoneSvGenderEmoSepKeSfMusic_test.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--start", type=int, default=0) # start from the last line number in jsonl where it stopped
    parser.add_argument("--tasks_to_infer", nargs="+", type=str, default=default_tasks)
    parser.add_argument("--split_audio", type=bool, default=False)
    # REQUIRED settings
    parser.add_argument("--write_path_title", type=str, default = None, required=True)
    parser.add_argument("--checkpoint_path", type=str, default=None, required=True) # it will automatically know which model_argument to use from this

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DEFAULT_WRITE_PATH_PARENT = "/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN/results"

    if args.debug:
        args.test_set_path = Path("/mnt/bn/audio-visual-llm-data/datasets/multitask_json/test_debug.json")
        DEFAULT_WRITE_PATH_PARENT = "/mnt/bn/audio-visual-llm-data6/terumi/SALMONNv1.1/logs/debug/"
        args.write_path_title = f"debug_{args.write_path_title}"
        args.batch_size = 2
        args.worker_num = 1
    
    write_path = Path(DEFAULT_WRITE_PATH_PARENT) / f"{args.write_path_title}.jsonl"

    write_path.parent.mkdir(parents = True, exist_ok = True)

    main(args.test_set_path, write_path, args.batch_size, args.checkpoint_path, args.max_new_tokens, start = args.start, task_filter = args.tasks_to_infer, split_audio=args.split_audio)
