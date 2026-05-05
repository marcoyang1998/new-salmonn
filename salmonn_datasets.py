import sys
import io
import json
import math
import random
import string
from typing import List, Dict

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from lhotse import Fbank, FbankConfig
from torchaudio.transforms import MelSpectrogram
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor, AutoProcessor
import soundfile as sf
PETRELOSS_CONFIG = "/mnt/shared-storage-user/housiyuan/xiaoyu/petreloss.conf"

def load_audio_from_petrel_oss(audio_path: str, client):
    bytes_data = client.get(audio_path)
    waveform, orig_sr = torchaudio.load(io.BytesIO(bytes_data))
    return waveform, orig_sr

class SALMONN_Dataset(Dataset):
    def __init__(self, args, tokenizer, encoder_type, llm_type):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        if encoder_type == "whisper_beats" or encoder_type == "audio_flamingo":
            self.max_frames = 30 * 16000
        else:
            self.max_frames = 120 * 16000 # TODO: make the longest accept duration configurable
        audio_chunk = getattr(args, "audio_chunk", 60)
        self.audio_chunk = audio_chunk * 16000 # TODO: make this optional
        self.split_audio = args.split_audio
        self.shuffle_mc_options = bool(getattr(args, "shuffle_mc_options", True))

        self.data = json.load(open(args.data_path, "r"))["data"]

        max_dur = getattr(args, "max_audio_duration", -1)
        if max_dur > 0:
            def _total_hours(entries):
                total_s = sum(
                    d
                    for e in entries
                    for d in e.get("durations", [])
                    if d >= 0
                )
                return total_s / 3600

            before_count = len(self.data)
            before_hours = _total_hours(self.data)
            self.data = [
                e for e in self.data
                if all(d < max_dur for d in e.get("durations", []) if d >= 0)
            ]
            after_count = len(self.data)
            after_hours = _total_hours(self.data)
            removed = before_count - after_count
            print(
                f"[Dataset] max_audio_duration={max_dur}s — "
                f"removed {removed:,} / {before_count:,} entries "
                f"({removed / before_count * 100:.1f}%)\n"
                f"  Total duration: {before_hours:.2f}h → {after_hours:.2f}h "
                f"(removed {before_hours - after_hours:.2f}h)",
                flush=True,
            )
        else:
            print(f"[Dataset] max_audio_duration disabled — keeping all {len(self.data):,} entries.", flush=True)

        min_dur = getattr(args, "min_audio_duration", 0.1)
        before_count = len(self.data)
        self.data = [
            e for e in self.data
            if all(d >= min_dur for d in e.get("durations", []) if d >= 0)
        ]
        removed = before_count - len(self.data)
        print(
            f"[Dataset] min_audio_duration={min_dur}s — "
            f"removed {removed:,} / {before_count:,} entries.",
            flush=True,
        )

        # lengths[i] = max audio duration (seconds) across all audios in sample i.
        # Used by AudioLengthGroupedSampler when group_by_audio_length=True.
        # Falls back to 0.0 for entries without a "durations" field.
        self.lengths = [
            max((d for d in e.get("durations", []) if d >= 0), default=0.0)
            for e in self.data
        ]

        self.encoder_type = encoder_type
        self.llm_type = llm_type
        if encoder_type == "zipformer2" or encoder_type == "spear_transformer":
            self.fbank = Fbank(FbankConfig(num_mel_bins=128))
        elif encoder_type == "dasheng":
            self.fbank = AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng", trust_remote_code=True)
        elif encoder_type == "whisper_beats" or encoder_type == "whisper":
            self.fbank = WhisperFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2")
        elif encoder_type == "qwenomni":
            self.fbank = WhisperFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B")
        elif encoder_type == "qwen3omni":
            self.fbank = WhisperFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-Omni-30B-A3B-Instruct")
        elif encoder_type == "mimo":
            self.fbank = MelSpectrogram(
                sample_rate=24000,
                n_fft=960,
                hop_length=240,
                win_length=960,
                power=1.0,
                center=True,
            )
        elif encoder_type == "perception_av":
            from core.audio_visual_encoder import PEAudioVisualTransform
            self.fbank = PEAudioVisualTransform.from_config("/mnt/bn/audio-visual-llm-data6/ckpts/pe-av-large", max_seconds = self.max_frames / 16000)
        elif encoder_type == "audio_flamingo":
            self.fbank = WhisperFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/audio-flamingo-3-hf")

        self.client = None
        self._broken_audio_paths = set()
        self.broken_sample_max_retries = max(1, int(getattr(args, "broken_sample_max_retries", 32)))

    def _load_audio(self, audio: str):
        if audio.startswith("s3://"):
            if self.client is None:
                from petrel_client.client import Client
                self.client = Client(PETRELOSS_CONFIG)
            return load_audio_from_petrel_oss(audio, self.client)
        else:
            return torchaudio.load(audio)
    
    def __len__(self):
        return len(self.data)

    def _option_label(self, index):
        if index < len(string.ascii_uppercase):
            return string.ascii_uppercase[index]
        return str(index + 1)

    def _build_mc_messages(self, sample):
        options = sample.get("mc_options")
        question = sample.get("mc_question")
        if options is None or question is None:
            return sample["messages"]

        if len(options) > 0 and isinstance(options[0], dict):
            if any("is_correct" in option for option in options):
                option_items = [
                    {
                        "content": option["content"],
                        "is_correct": bool(option.get("is_correct", False)),
                    }
                    for option in options
                ]
            else:
                answer_label = sample.get("mc_answer_label", "")
                option_items = [
                    {
                        "content": option["content"],
                        "is_correct": option.get("label", "") == answer_label,
                    }
                    for option in options
                ]
        else:
            answer_index = int(sample.get("mc_answer_index", -1))
            option_items = [
                {
                    "content": option_text,
                    "is_correct": idx == answer_index,
                }
                for idx, option_text in enumerate(options)
            ]

        if self.shuffle_mc_options:
            random.shuffle(option_items)

        lines = [
            "<audio>Listen to the audio and answer the following multiple-choice question.",
            f"Question: {question.strip()}",
            "Choices:",
        ]
        answer_label = None
        for idx, option in enumerate(option_items):
            label = self._option_label(idx)
            lines.append(f"Option {label}: {option['content']}")
            if option["is_correct"]:
                answer_label = label
        lines.append("")
        lines.append('Please output your final answer with a single letter. For example, if you think the answer is Option A, please just output \'A\'.')

        if answer_label is None:
            raise ValueError("Cannot determine correct answer for multiple-choice sample.")

        return [
            {
                "role": "user",
                "content": "\n".join(lines),
            },
            {
                "role": "assistant",
                "content": answer_label,
            },
        ]
    
    def _verify_chat(self, chats: List[Dict]):
        assert len(chats) == 2, f"Chat should have exactly 2 messages, but got {len(chats)}: {chats}"
        assert "<audio>" in chats[0]["content"], f"The first message should contain the audio placeholder, but got: {chats}"
    
    def __getitem__(self, index):
        for attempt in range(self.broken_sample_max_retries):
            sample_index = index if attempt == 0 else random.randint(0, len(self.data) - 1)
            sample = self.data[sample_index]
            try:
                result = self._load_sample(sample)
                return result
            except Exception as e:
                for audio_path in sample.get("audios", []):
                    if audio_path not in self._broken_audio_paths:
                        print(f"[WARN] Broken audio detected and skipped: {audio_path}; error={repr(e)}", flush=True)
                        self._broken_audio_paths.add(audio_path)
        raise RuntimeError(
            f"Failed to fetch a valid sample after {self.broken_sample_max_retries} retries due to broken audio files."
        )

    def _load_sample(self, sample):
        fbanks = []
        fbank_lens = []
        raw_wavs = []
        audio_nums = []
        for audio_path in sample["audios"]:
            audio, fs = self._load_audio(audio_path)
            # some of our audio is not 16k hz, so we first resample them and then truncate it
            if fs != 16000:
                audio = torchaudio.functional.resample(audio, fs, 16000)
                fs = 16000
            if audio.shape[1] > self.max_frames:
                audio = audio[:, :self.max_frames]
            assert fs == 16000
            if self.encoder_type == "zipformer2" or self.encoder_type == "spear_transformer":
                if self.split_audio and audio.shape[-1] > self.audio_chunk:
                    if audio.size(0) > 1:
                        audio = audio.mean(dim=0, keepdim=True)
                    pad_len = (-audio.shape[-1]) % self.audio_chunk
                    if pad_len > 0:
                        audio = F.pad(audio, (0, pad_len))
                    audio = audio.unfold(-1, self.audio_chunk, self.audio_chunk)[0]
                    item_fbanks = self.fbank.extract_batch(audio, sampling_rate=fs)
                    if item_fbanks.ndim == 2:
                        item_fbanks = item_fbanks.unsqueeze(0)
                    audio_nums.append(item_fbanks.size(0))
                    for item_fbank in item_fbanks:
                        fbanks.append(item_fbank)
                        fbank_lens.append(fbanks[-1].size(0))
                    fbank_lens[-1] = max(math.ceil((self.audio_chunk - pad_len) / 160),50)
                else:
                    fbanks.append(self.fbank.extract(audio.squeeze(), sampling_rate=fs))
                    fbank_lens.append(fbanks[-1].size(0))
            elif self.encoder_type == "dasheng":
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                fbanks.append(self.fbank(audio, sampling_rate=fs, return_tensors="pt").input_values.squeeze(0).transpose(0, 1))
                fbank_lens.append(fbanks[-1].size(0))
            elif self.encoder_type == "dasheng_wavlm":
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                fbanks.append(audio.squeeze())
                fbank_lens.append(fbanks[-1].size(0))
            elif self.encoder_type == "whisper_beats":
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                raw_wavs.append(audio.squeeze())
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                fbanks.append(self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
                fbank_lens.append(raw_wavs[-1].size(0))
            elif self.encoder_type == "whisper":
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                fbanks.append(self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
                fbank_lens.append(1500)
            elif self.encoder_type == "qwenomni" or self.encoder_type == "qwen3omni":
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                one_fbank = self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
                fbanks.append(one_fbank["input_features"].squeeze())
                raw_wavs.append(one_fbank["attention_mask"].squeeze())
                fbank_lens.append(one_fbank["attention_mask"].sum(-1))
            elif self.encoder_type == "mimo":
                if audio.ndim == 2:
                    audio = audio.mean(dim=0)
                audio = torchaudio.functional.resample(audio, fs, 24000)
                spec = self.fbank(audio[None, :])
                mel = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)
                fbanks.append(mel)
                fbank_lens.append(mel.size(0))
            elif self.encoder_type == "perception_av":
                inputs = self.fbank(videos=None, audio=[audio_path], text=None)
                fbanks.append(inputs['input_values'].squeeze())
                raw_wavs.append(inputs['padding_mask'].squeeze())
                fbank_lens.append(inputs['padding_mask'].sum(-1))
            elif self.encoder_type == "audio_flamingo":
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                inputs = self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
                fbanks.append(inputs["input_features"].squeeze())
                raw_wavs.append(inputs["attention_mask"].squeeze())
                fbank_lens.append(inputs["attention_mask"].sum(-1))
        
        if sample.get("task_type") == "qa_mc" and "mc_options" in sample and "mc_question" in sample:
            chats = self._build_mc_messages(sample)
        else:
            chats = sample["messages"]
        
        chats = sample["messages"]
        self._verify_chat(chats)
        text = self.tokenizer.apply_chat_template(chats,tokenize=False)
        if self.llm_type == "Qwen":
            if self.split_audio and len(audio_nums) > 0:
                for audio_num in audio_nums:
                    text = text.replace("<audio>","<|vision_start|>"*audio_num+"<|vision_end|>",1)
                model_inputs = self.tokenizer(text, return_tensors="pt")
            else:
                model_inputs = self.tokenizer(
                    text.replace("<audio>","<|vision_start|><|vision_end|>"), return_tensors="pt"
                )
        elif self.llm_type == "Llama":
            model_inputs = self.tokenizer(text.replace("<audio>","<|reserved_special_token_0|><|reserved_special_token_1|>"), return_tensors="pt", add_special_tokens=False)
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0]

        labels = torch.full((input_ids.shape[-1],), fill_value=-100, dtype=torch.long)
        if self.llm_type == "Qwen":
            im_start_id = 151644
            im_end_id = 151645
            assistant_id = 77091
            shift_num = 3
        elif self.llm_type == "Llama":
            im_start_id = 128006
            im_end_id = 128009
            assistant_id = 78191
            shift_num = 4
        im_start_idx = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
        im_end_idx = (input_ids == im_end_id).nonzero(as_tuple=True)[0]
        for idx, end_idx in zip(im_start_idx, im_end_idx):
            if input_ids[idx+1] == assistant_id:
                labels[idx+shift_num:end_idx+1] = input_ids[idx+shift_num:end_idx+1]

        new_sample = {}
        new_sample["input_ids"] = input_ids
        new_sample["attention_mask"] = attention_mask
        new_sample["labels"] = labels
        new_sample["fbank_feature"] = fbanks
        new_sample["fbank_feature_len"] = fbank_lens
        new_sample["raw_wavs"] = raw_wavs

        return new_sample