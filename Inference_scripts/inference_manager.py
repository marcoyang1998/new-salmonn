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
from pdb import set_trace as st

TOKENIZER_PATH="/mnt/shared-storage-user/brainllm-share/checkpoints/Qwen3-8B"
override_keys = ["weighted_sum_encoder", "concat_encoder_features"]

logging.getLogger().setLevel(logging.INFO)

def override_args(checkpoint_path: str, default_model_args):
    config_file = os.path.dirname(checkpoint_path) + "/config.json"
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
    def __init__(self, checkpoint_path: str, max_new_tokens=500, device=0, task_filter=None, split_audio: bool = False):
        self.model_args = get_model_args(checkpoint_path)
        self.model_args = override_args(checkpoint_path, self.model_args)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.task_filter = task_filter
        self.split_audio = split_audio
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        self.fbank = get_fbank(self.model_args)
        self.model = self._load_model()

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
                messages, tokenize=False, add_generation_prompt=True
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

if __name__ == "__main__":
    checkpoint_path = "output/test_stage2_step30000_120s_spear_xlarge_token_mix_bf16_weighted_sum/checkpoint-10000"
    engine = InferenceManager(checkpoint_path=checkpoint_path, device="cuda")
    model = engine.model
    # print(model.audio_encoder_layer_weights)
    # print(model.audio_encoder_layer_weights.softmax(dim=-1))
    print("pass")