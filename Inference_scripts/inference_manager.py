from dataclasses import dataclass, field, asdict
from modeling_salmonn import SALMONN
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
import os
import json
from tqdm import tqdm
from inference_utils import get_prompt, get_audio_path_list, extract_audio_features, get_fbank, prepare_model_inputs, get_model_args
import logging

TOKENIZER_PATH="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/models/Qwen3-8B"

override_keys = [
    "weighted_sum_encoder",
    "concat_encoder_features",
    "zipformer_version",
    "connector_hid_size",
    "connector_seg_size",
    "connector_type"
]

def override_args(checkpoint_path: str, default_model_args):
    config_file = os.path.dirname(checkpoint_path) + "/config.json"
    if not os.path.exists(config_file):
        logging.warning(f"Config file {config_file} not found. Using default model args without override.")
        return default_model_args
    with open(config_file, "r") as f:
        config = json.load(f)
        model_args = config["model_args"]
    # we pre-define some keys to be overriden
    for k in override_keys:
        attr = model_args.get(k, None)
        if attr is not None:
            setattr(default_model_args, k, attr)
            print(f"Setting {k} to {attr} as specified in the checkpoint config.")
    
    return default_model_args

class InferenceManager:
    def __init__(
        self,
        checkpoint_path: str,
        max_new_tokens=500,
        device=0,
        task_filter=None,
        split_audio: bool = False,
        disable_thinking: bool=True,
        tokenizer_path: str = TOKENIZER_PATH
    ):
        self.model_args = get_model_args(checkpoint_path)
        self.model_args = override_args(checkpoint_path, self.model_args)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.task_filter = task_filter
        self.split_audio = split_audio
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.fbank = get_fbank(self.model_args)
        self.model = self._load_model()
        self.disable_thinking = disable_thinking

    def _load_model(self):
        model = SALMONN.from_pretrained(
            self.model_args.model_name_or_path,
            config=AutoConfig.from_pretrained(os.path.join(self.model_args.model_name_or_path, "config.json")),
            model_args=self.model_args,
            torch_dtype="auto",
            device_map=self.device
        )
        model.eval()
        return model

    def process_batch(self, batch_data, out_file):
        original_batch_size = len(batch_data)
        if self.task_filter:
            batch_data = [sample for sample in batch_data if sample.get("task") in self.task_filter]
            if not batch_data:
                return
        texts = []
        audio_paths = []
        for sample in batch_data:
            prompt = get_prompt(sample)
            audio_path_list = get_audio_path_list(sample)
            audio_num = len(audio_path_list)
            audio_paths.extend(audio_path_list)
            messages = [{"role": "user", "content": "<audio>" * audio_num + prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=not self.disable_thinking,
            )
            texts.append(text)

        features, feature_lens, raw_wavs, audio_nums = extract_audio_features(audio_paths, self.fbank, self.model, self.model_args, split_audio=self.split_audio)

        model_inputs = prepare_model_inputs(texts, audio_nums, self.tokenizer, self.model)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                fbank_feature=features,
                fbank_feature_len=feature_lens,
                raw_wavs=raw_wavs,
                max_new_tokens=self.max_new_tokens,
            )

        for i, generated_id in enumerate(generated_ids):
            output_ids = generated_id.tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            cleaned_content = content.split("</think>")[-1].strip()
            sample = batch_data[i]
            sample["response"] = cleaned_content
            json.dump(sample, out_file, ensure_ascii=False)
            out_file.write("\n")
