from dataclasses import dataclass, field, asdict
from modeling_salmonn import SALMONN
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
import os
import json
from tqdm import tqdm
from inference_utils import get_prompt, get_audio_path_list, extract_audio_features, get_fbank, maybe_init_qwen3_embedding_model, prepare_model_inputs, get_model_args, OVERRIDE_KEYS, override_args_from_config
import logging

TOKENIZER_PATH="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/models/Qwen3-8B"


class InferenceManager:
    def __init__(
        self,
        checkpoint_path: str,
        max_new_tokens=500,
        device=0,
        task_filter=None,
        split_audio: bool = False,
        disable_thinking: bool=True,
        tokenizer_path: str = TOKENIZER_PATH,
        use_beam_search: bool = False,
        beam_size: int = 4,
        use_nucleus_sampling: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.8,
        top_k: int = 20,
        min_p: float = 0.0,
        seed: int = 42,
    ):
        self.model_args = get_model_args(checkpoint_path)
        self.model_args = override_args_from_config(checkpoint_path, self.model_args)
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.task_filter = task_filter
        self.split_audio = split_audio
        self.use_beam_search = use_beam_search
        self.beam_size = beam_size
        self.use_nucleus_sampling = use_nucleus_sampling
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_p = min_p
        self.seed = seed
        # TODO: check if using checkpoint_path is safe
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.fbank = get_fbank(self.model_args)
        self.model = self._load_model()
        self.disable_thinking = disable_thinking

    def _load_model(self):
        model = SALMONN.from_pretrained(
            self.model_args.model_name_or_path,
            config=AutoConfig.from_pretrained(os.path.join(self.model_args.model_name_or_path, "config.json")),
            model_args=self.model_args,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        if self.model_args.inject_temporal_embedding:
            model.register_temporal_tokens(self.tokenizer)
        if getattr(self.model_args, "inject_temporal_embedding_nl", False):
            model.register_nl_timestamp_tokenizer(self.tokenizer)
        maybe_init_qwen3_embedding_model(model, self.model_args)
        model.eval()
        return model

    def process_batch(self, batch_data, out_file):
        original_batch_size = len(batch_data)
        if self.task_filter:
            batch_data = [sample for sample in batch_data if sample.get("task") in self.task_filter]
            if not batch_data:
                return
        texts = []
        user_prompts = []
        audio_paths = []
        for sample in batch_data:
            prompt = get_prompt(sample)
            user_prompts.append(prompt)
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

        if self.use_beam_search:
            generation_config = {
                "num_beams": self.beam_size,
                "early_stopping": True,
                "do_sample": False,
            }
        elif self.use_nucleus_sampling:
            generation_config = {
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "min_p": self.min_p,
            }
        else:
            generation_config = {}
        with torch.no_grad():
            # Fix the seed right before generation for deterministic nucleus sampling.
            torch.manual_seed(self.seed)
            generated_ids = self.model.generate(
                **model_inputs,
                fbank_feature=features,
                fbank_feature_len=feature_lens,
                raw_wavs=raw_wavs,
                user_prompts=user_prompts,
                max_new_tokens=self.max_new_tokens,
                **generation_config,
            )

        for i, generated_id in enumerate(generated_ids):
            output_ids = generated_id.tolist()
            content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
            cleaned_content = content.split("</think>")[-1].strip()
            # TODO: fix this bug
            if cleaned_content.startswith("<think>\n\n"):
                cleaned_content = cleaned_content.replace("<think>\n\n", "", 1).strip()
            sample = batch_data[i]
            sample["response"] = cleaned_content
            json.dump(sample, out_file, ensure_ascii=False)
            out_file.write("\n")
