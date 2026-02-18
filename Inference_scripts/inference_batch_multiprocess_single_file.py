from dataclasses import dataclass, field, asdict
from modeling_salmonn import SALMONN
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
import os
import json
from tqdm import tqdm
from inference_utils import get_prompt, get_audio_path_list, extract_audio_features, get_fbank, prepare_model_inputs
from torch.multiprocessing import Process, set_start_method
import logging
from pathlib import Path
from argparse import ArgumentParser
from inference_manager import InferenceManager
from pdb import set_trace as st

logging.basicConfig(
    filename = 'logs/inference_batch_multiprocess.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def worker(worker_id: int, data_chunk, batch_size, out_file_path, max_new_tokens, checkpoint_path: str, task_filter: list[str] = None):
    torch.cuda.set_device(worker_id)
    out_file_path.parent.mkdir(parents=True, exist_ok=True)
    inference_manager = InferenceManager(checkpoint_path, max_new_tokens, device=worker_id, task_filter=task_filter)
    
    with open(out_file_path, "a", encoding="utf-8") as out_file:
        batch_data = []
        for i, sample in enumerate(tqdm(data_chunk, desc=f"Worker {worker_id} Inferencing")):
            batch_data.append(sample)
            if len(batch_data) == batch_size:
                inference_manager.process_batch(batch_data, out_file)
                batch_data = []
                print(f"Worker {worker_id} processed {i + 1} samples")
                out_file.flush()
        if batch_data:
            inference_manager.process_batch(batch_data, out_file)
            print(f"processed {len(data_chunk)} samples")

def main(test_set_path, write_path: Path, batch_size, worker_num, checkpoint_path: str, max_new_tokens: int = 500, start: int = 0, task_filter: list = None):
    set_start_method("spawn", force=True)
    with open(test_set_path, "r") as f:
        data = json.load(f)["annotation"]
    
    data = data[start:]
    data_chunks = [data[i::worker_num] for i in range(worker_num)]

    prefix = write_path.stem.split("_", 1)[0]
    temp_dir = write_path.parent / f"{prefix}_temp_dir"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_paths = [(temp_dir / f"temp_part{i}{write_path.suffix}") for i in range(worker_num)]
    processes = []
    for worker_id in range(worker_num):
        p = Process(target=worker, args=(worker_id, data_chunks[worker_id], batch_size, temp_paths[worker_id], max_new_tokens, checkpoint_path, task_filter))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    
    with open(write_path, "w", encoding="utf-8") as out_file:
        final_results = [None] * len(data)
        for worker_id, temp_path in enumerate(temp_paths):
            write_indices = [i for i in range(worker_id, len(data), worker_num)]
            with open(temp_path, "r", encoding="utf-8") as temp_file:
                for i, line in enumerate(temp_file):
                    if i < len(write_indices):
                        sample = json.loads(line)
                        final_results[write_indices[i]] = sample
                    else:
                        logging.error(f"More lines in temp file {temp_path} than expected. This means some data are missing during inference to the temp file!")

        for sample in final_results:
            if sample is not None:
                json.dump(sample, out_file, ensure_ascii=False)
                out_file.write("\n")
            else:
                logging.error("FINAL FILE OUTPUT: Some samples are missing in the final results!")

    # 删除临时文件
    for temp_path in temp_paths:
        if temp_path.exists():
            temp_path.unlink()

    # 删除临时目录
    if temp_dir.exists():
        temp_dir.rmdir()

if __name__ == "__main__":
    default_tasks = [
                     'keywords',
                     'audiocaption', 
                     'asr', 
                     'translation_ec', 
                     'phone_recognition', 
                     'speech_query', 
                     'speaker_verification', 
                     'emotion_recognition', 
                     'gender_QA', 
                     'speech_separation', 
                     'slot_filling', 
                     'music_description'
                     ]


    parser = ArgumentParser()
    # with default settings
    parser.add_argument("--test_set_path", type=str, default = "/mnt/bn/audio-visual-llm-data/datasets/multitask_json/test_AsrAacAstQueryPhoneSvGenderEmoSepKeSfMusic_test.json")
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--worker_num", type=int, default=4) # currently equivalent to num of workers, each gpu taking one worker
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--start", type=int, default=0) # start from the last line number in jsonl where it stopped
    parser.add_argument("--tasks_to_infer", type=list[str], default=default_tasks)
    # REQUIRED settings
    parser.add_argument("--write_path_title", type=str, default = None, required=True) # title cannot contain "_"
    parser.add_argument("--checkpoint_path", type=str, default=None, required=True) # it will automatically know which model_argument to use from this

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    DEFAULT_WRITE_PATH_PARENT = "/mnt/bn/audio-visual-llm-data6/terumi/SALMONNv1.1_eval/Inference_data"

    if args.debug:
        args.test_set_path = Path("/mnt/bn/audio-visual-llm-data/datasets/multitask_json/test_debug.json")
        DEFAULT_WRITE_PATH_PARENT = "/mnt/bn/audio-visual-llm-data6/terumi/SALMONNv1.1/logs/debug/"
        args.write_path_title = f"debug_{args.write_path_title}"
        args.batch_size = 2
        args.worker_num = 1

    write_path = Path(DEFAULT_WRITE_PATH_PARENT) / f"{args.write_path_title}_MultP_Batch_AsrAacAstQueryPhoneSvGenderEmoSepKeSfMusic.jsonl"

    write_path.parent.mkdir(parents = True, exist_ok = True)

    main(args.test_set_path, write_path, args.batch_size, args.worker_num, args.checkpoint_path, max_new_tokens = args.max_new_tokens, start = args.start, task_filter = args.tasks_to_infer)