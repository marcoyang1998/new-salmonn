import sys
import io
import json
import math
import random
import string
import copy
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
        self.skip_thinking_token_loss = bool(getattr(args, "skip_thinking_token_loss", False))
        self.ctx_biasing_list_min_ratio = float(getattr(args, "ctx_biasing_list_min_ratio", 0.5))
        self.ctx_biasing_list_max_ratio = float(getattr(args, "ctx_biasing_list_max_ratio", 1.0))
        if not 0 < self.ctx_biasing_list_min_ratio <= self.ctx_biasing_list_max_ratio <= 1:
            raise ValueError(
                "ctx_biasing_list_min_ratio and ctx_biasing_list_max_ratio must satisfy "
                f"0 < min <= max <= 1, but got {self.ctx_biasing_list_min_ratio} and {self.ctx_biasing_list_max_ratio}."
            )

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

        min_dur = getattr(args, "min_audio_duration", 0.3)
        self.min_audio_duration = min_dur
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
        assert chats[0]["content"].startswith("<audio>"), f"The first message should start with the audio placeholder, but got: {chats}"

    def _get_user_prompt(self, chats: List[Dict]) -> str:
        return chats[0]["content"].replace("<audio>", " ").strip()

    def _get_valid_think_paths(self, sample: Dict) -> List[str]:
        think_paths = sample.get("think_paths", [])
        if not isinstance(think_paths, list):
            return []
        return [
            path.strip()
            for path in think_paths
            if isinstance(path, str) and path.strip()
        ]

    def _sample_think_path(self, sample: Dict) -> str:
        think_paths = self._get_valid_think_paths(sample)
        return random.choice(think_paths) if think_paths else ""

    def _has_reasoning_content(self, chats: List[Dict]) -> bool:
        return any(
            isinstance(chat, dict)
            and chat.get("role") == "assistant"
            and isinstance(chat.get("reasoning_content"), str)
            and chat["reasoning_content"].strip()
            for chat in chats
        )

    def _build_contextual_asr_messages(self, sample: Dict) -> List[Dict]:
        chats = copy.deepcopy(sample["messages"])
        biasing_list = sample.get("biasing_list", [])
        if not chats or not isinstance(chats[0], dict):
            return chats
        if not biasing_list:
            return chats

        think_path = self._sample_think_path(sample)
        biasing_words = [str(word) for word in biasing_list]
        if not think_path:
            if len(biasing_words) > 10:
                sample_size = random.randint(10, min(20, len(biasing_words)))
                biasing_words = random.sample(biasing_words, sample_size)
            else:
                random.shuffle(biasing_words)
        biasing_text = f"[{', '.join(biasing_words)}]"
        if not biasing_text:
            return chats

        user_content = chats[0].get("content", "")
        user_content = user_content.rstrip()
        if user_content and user_content[-1] not in ".!?:":
            user_content += "."
        if user_content:
            user_content += "\n"
        user_content += (
            "Pay extra attention to the following contextual words:\n"
            "<biasing_list>\n"
            f"{biasing_text}\n"
            "</biasing_list>."
        )
        chats[0]["content"] = user_content
        if (
            think_path
            and len(chats) > 1
            and isinstance(chats[1], dict)
            and chats[1].get("role") == "assistant"
        ):
            chats[1]["reasoning_content"] = think_path
        return chats

    def _build_multimodal_contextual_asr_messages(self, sample: Dict) -> List[Dict]:
        chats = copy.deepcopy(sample["messages"])
        biasing_list = sample.get("biasing_list", [])
        ctx_audios = sample.get("ctx_audios", [])
        if not chats or not isinstance(chats[0], dict):
            return chats
        if not isinstance(biasing_list, list) or not isinstance(ctx_audios, list):
            raise ValueError("Multimodal contextual ASR requires list fields: biasing_list and ctx_audios.")
        if len(biasing_list) != len(ctx_audios):
            raise ValueError(
                "Multimodal contextual ASR requires one ctx_audio per biasing word, "
                f"but got {len(ctx_audios)} ctx_audios and {len(biasing_list)} biasing words."
            )
        if not biasing_list:
            return chats

        user_content = chats[0].get("content", "").rstrip()
        if user_content and user_content[-1] not in ".!?:":
            user_content += "."
        if user_content:
            user_content += "\n"
        lines = [
            user_content + "Use the following contextual words and their pronunciations as references while transcribing the speech:",
            "<biasing_list>",
        ]
        for word in biasing_list:
            lines.append(f"<audio>{word}")
        lines.append("</biasing_list>.")
        chats[0]["content"] = "\n".join(lines)
        return chats

    def _sample_multimodal_contextual_asr_biasing(self, sample: Dict) -> Dict:
        if sample.get("task_type") != "contextual_ASR" or "ctx_audios" not in sample:
            return sample
        if sample.get("_ctx_audios_sampled", False):
            return sample
        biasing_list = sample.get("biasing_list", [])
        ctx_audios = sample.get("ctx_audios", [])
        if not isinstance(biasing_list, list) or not isinstance(ctx_audios, list):
            raise ValueError("Multimodal contextual ASR requires list fields: biasing_list and ctx_audios.")
        if len(biasing_list) != len(ctx_audios):
            raise ValueError(
                "Multimodal contextual ASR requires one ctx_audio per biasing word, "
                f"but got {len(ctx_audios)} ctx_audios and {len(biasing_list)} biasing words."
            )
        if len(biasing_list) <= 1:
            return sample

        min_len = max(1, math.ceil(len(biasing_list) * self.ctx_biasing_list_min_ratio))
        max_len = max(min_len, math.floor(len(biasing_list) * self.ctx_biasing_list_max_ratio))
        sample_size = random.randint(min_len, max_len)
        sampled_pairs = random.sample(list(zip(biasing_list, ctx_audios)), sample_size)

        sampled_sample = copy.deepcopy(sample)
        sampled_sample["biasing_list"] = [word for word, _ in sampled_pairs]
        sampled_sample["ctx_audios"] = [ctx_audio for _, ctx_audio in sampled_pairs]
        sampled_sample["_ctx_audios_sampled"] = True
        return sampled_sample

    def _get_sample_audio_paths(self, sample: Dict) -> List[str]:
        audio_paths = list(sample["audios"])
        if sample.get("task_type") == "contextual_ASR" and "ctx_audios" in sample:
            ctx_audios = sample.get("ctx_audios", [])
            if not isinstance(ctx_audios, list):
                raise ValueError("ctx_audios must be a list when present.")
            audio_paths.extend(ctx_audios)
        return audio_paths

    def _get_grouped_sample_audio_paths(self, sample: Dict):
        main_audio_paths = list(sample["audios"])
        ctx_audio_paths = []
        if sample.get("task_type") == "contextual_ASR" and "ctx_audios" in sample:
            ctx_audio_paths = sample.get("ctx_audios", [])
            if not isinstance(ctx_audio_paths, list):
                raise ValueError("ctx_audios must be a list when present.")
        return [(audio_path, 0) for audio_path in main_audio_paths] + [(audio_path, 1) for audio_path in ctx_audio_paths]
    
    def __getitem__(self, index):
        for attempt in range(self.broken_sample_max_retries):
            sample_index = index if attempt == 0 else random.randint(0, len(self.data) - 1)
            sample = self._sample_multimodal_contextual_asr_biasing(self.data[sample_index])
            try:
                result = self._load_sample(sample)
                return result
            except Exception as e:
                for audio_path in self._get_sample_audio_paths(sample):
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
        ctx_fbanks = []
        ctx_fbank_lens = []
        ctx_raw_wavs = []
        audio_nums = []
        audio_feature_groups = []
        audio_files = []
        for audio_path, audio_group in self._get_grouped_sample_audio_paths(sample):
            target_fbanks = fbanks if audio_group == 0 else ctx_fbanks
            target_fbank_lens = fbank_lens if audio_group == 0 else ctx_fbank_lens
            target_raw_wavs = raw_wavs if audio_group == 0 else ctx_raw_wavs
            audio, fs = self._load_audio(audio_path)
            duration = audio.shape[1] / fs
            if duration < self.min_audio_duration:
                print(
                    f"[WARN] dAudio too short ({duration:.3f}s < min_audio_duration={self.min_audio_duration}s), "
                    f"skipping and resampling: {audio_path}",
                    flush=True,
                )
                raise ValueError(f"Audio duration {duration:.3f}s below minimum {self.min_audio_duration}s")
            audio_files.append(audio_path)
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
                    audio_feature_groups.extend([audio_group] * item_fbanks.size(0))
                    for item_fbank in item_fbanks:
                        target_fbanks.append(item_fbank)
                        target_fbank_lens.append(target_fbanks[-1].size(0))
                    target_fbank_lens[-1] = max(math.ceil((self.audio_chunk - pad_len) / 160),50)
                else:
                    audio_nums.append(1)
                    audio_feature_groups.append(audio_group)
                    target_fbanks.append(self.fbank.extract(audio.squeeze(), sampling_rate=fs))
                    target_fbank_lens.append(target_fbanks[-1].size(0))
            elif self.encoder_type == "dasheng":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                target_fbanks.append(self.fbank(audio, sampling_rate=fs, return_tensors="pt").input_values.squeeze(0).transpose(0, 1))
                target_fbank_lens.append(target_fbanks[-1].size(0))
            elif self.encoder_type == "dasheng_wavlm":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                target_fbanks.append(audio.squeeze())
                target_fbank_lens.append(target_fbanks[-1].size(0))
            elif self.encoder_type == "whisper_beats":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                target_raw_wavs.append(audio.squeeze())
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                target_fbanks.append(self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
                target_fbank_lens.append(target_raw_wavs[-1].size(0))
            elif self.encoder_type == "whisper":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                target_fbanks.append(self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
                target_fbank_lens.append(1500)
            elif self.encoder_type == "qwenomni" or self.encoder_type == "qwen3omni":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                one_fbank = self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
                target_fbanks.append(one_fbank["input_features"].squeeze())
                target_raw_wavs.append(one_fbank["attention_mask"].squeeze())
                target_fbank_lens.append(one_fbank["attention_mask"].sum(-1))
            elif self.encoder_type == "mimo":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                if audio.ndim == 2:
                    audio = audio.mean(dim=0)
                audio = torchaudio.functional.resample(audio, fs, 24000)
                spec = self.fbank(audio[None, :])
                mel = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)
                target_fbanks.append(mel)
                target_fbank_lens.append(mel.size(0))
            elif self.encoder_type == "perception_av":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                inputs = self.fbank(videos=None, audio=[audio_path], text=None)
                target_fbanks.append(inputs['input_values'].squeeze())
                target_raw_wavs.append(inputs['padding_mask'].squeeze())
                target_fbank_lens.append(inputs['padding_mask'].sum(-1))
            elif self.encoder_type == "audio_flamingo":
                audio_nums.append(1)
                audio_feature_groups.append(audio_group)
                sf_audio, _ = sf.read(audio_path, frames=self.max_frames)
                if len(sf_audio.shape) == 2: # stereo to mono
                    sf_audio = sf_audio[:, 0]
                inputs = self.fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
                target_fbanks.append(inputs["input_features"].squeeze())
                target_raw_wavs.append(inputs["attention_mask"].squeeze())
                target_fbank_lens.append(inputs["attention_mask"].sum(-1))
        
        if sample.get("task_type") == "qa_mc" and "mc_options" in sample and "mc_question" in sample:
            chats = self._build_mc_messages(sample)
        elif sample.get("task_type") == "contextual_ASR" and "ctx_audios" in sample:
            chats = self._build_multimodal_contextual_asr_messages(sample)
        elif sample.get("task_type") == "contextual_ASR":
            chats = self._build_contextual_asr_messages(sample)
        else:
            chats = sample["messages"]
        has_reasoning_content = self._has_reasoning_content(chats)
        
        self._verify_chat(chats)
        audio_placeholder_count = chats[0]["content"].count("<audio>")
        if audio_placeholder_count != len(audio_nums):
            raise ValueError(
                f"Audio placeholder count ({audio_placeholder_count}) does not match loaded audio count "
                f"({len(audio_nums)}): {audio_files}"
            )
        if len(audio_feature_groups) != sum(audio_nums):
            raise ValueError(
                f"Audio feature group count ({len(audio_feature_groups)}) does not match expanded audio count "
                f"({sum(audio_nums)}): {audio_files}"
            )
        text = self.tokenizer.apply_chat_template(chats,tokenize=False)
        if self.llm_type == "Qwen":
            for audio_num in audio_nums:
                text = text.replace("<audio>","<|vision_start|>"*audio_num+"<|vision_end|>",1)
            model_inputs = self.tokenizer(text, return_tensors="pt")
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
            if self.skip_thinking_token_loss and not has_reasoning_content:
                shift_num = 3 + 4
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
        new_sample["ctx_fbank_feature"] = ctx_fbanks
        new_sample["ctx_fbank_feature_len"] = ctx_fbank_lens
        new_sample["ctx_raw_wavs"] = ctx_raw_wavs
        new_sample["audio_feature_groups"] = audio_feature_groups
        new_sample["user_prompt"] = [self._get_user_prompt(chats)]
        new_sample["audio_files"] = audio_files

        return new_sample