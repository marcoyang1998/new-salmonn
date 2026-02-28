import sys
import json
import math
import random

import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from lhotse import Fbank, FbankConfig
from torchaudio.transforms import MelSpectrogram
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor, AutoProcessor
import soundfile as sf

class SALMONN_Dataset(Dataset):
    def __init__(self, args, tokenizer, encoder_type, llm_type):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        if encoder_type == "whisper_beats" or encoder_type == "audio_flamingo":
            self.max_frames = 30 * 16000
        else:
            self.max_frames = 120 * 16000
        self.audio_chunk = 120 * 16000 # We set a longer audio chunk for Zipformer
        self.split_audio = args.split_audio

        self.data = json.load(open(args.data_path, "r"))["data"]
        self.encoder_type = encoder_type
        self.llm_type = llm_type
        if encoder_type == "zipformer2":
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]

        fbanks = []
        fbank_lens = []
        raw_wavs = []
        audio_nums = []
        for audio_path in sample["audios"]:
            audio, fs = torchaudio.load(audio_path, num_frames=self.max_frames)
            if fs != 16000:
                audio = torchaudio.functional.resample(audio, fs, 16000)
                fs = 16000
            if self.encoder_type == "zipformer2":
                if self.split_audio:
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
        
        chats = sample["messages"]
        text = self.tokenizer.apply_chat_template(chats,tokenize=False)
        if self.llm_type == "Qwen":
            if self.split_audio:
                for audio_num in audio_nums:
                    text = text.replace("<audio>","<|vision_start|>"*audio_num+"<|vision_end|>",1)
                model_inputs = self.tokenizer(text, return_tensors="pt")
            else:
                model_inputs = self.tokenizer(text.replace("<audio>","<|vision_start|><|vision_end|>"), return_tensors="pt")
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