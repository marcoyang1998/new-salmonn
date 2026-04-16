import sys
import io
import json
import math
import random
import string
from typing import List, Dict

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.data import Dataset
from lhotse import Fbank, FbankConfig
from torchaudio.transforms import MelSpectrogram
from transformers import AutoFeatureExtractor, WhisperFeatureExtractor, AutoProcessor
import soundfile as sf
from petrel_client.client import Client


short_captions = [
    "Please describe the audio.",
    "Based on the sound you hear, create a caption for this audio.",
    "What does this audio describe?",
    "Describe the following audio in a caption.",
    "Listen to this audio clip and provide its caption.",
    "Could you summarise what's happening in this audio?",
    "Can you describe the scene or event depicted in this audio?",
    "Provide a short caption for this audio clip.",
]

long_captions = [
    "Describe the audio in detail.",
    "Provide a detailed description of the audio, covering any speech, background noises, or musical elements present.",
    "Provide a richly detailed caption for this audio file, detailing everything from vocal delivery and language to environmental sounds.",
    "Listen to the audio and provide a detailed caption, including any speech or background elements.",
    "Listen carefully to this recording and write a highly detailed caption capturing all sound events or human speech.",
    "What is happening in this audio? Please provide a thorough and detailed explanation.",
    "What do you hear in this recording? Please elaborate in detail on the voices, emotions, background elements, and the overall soundscape.",
    "Write a thorough description of the audio, capturing the complete sonic picture, including speech patterns, musical instruments, and acoustic artifacts.",
    "Describe the audio comprehensively, paying close attention to who is speaking, how they sound, and what else is happening in the recording.",
]


PETREL_CFG_PATH = "/mnt/shared-storage-user/xuxuenan/petreloss.conf"

class SALMONN_Dataset(Dataset):
    def __init__(self, args, tokenizer, encoder_type, llm_type):
        super().__init__()

        self.args = args
        self.tokenizer = tokenizer
        if encoder_type == "whisper_beats" or encoder_type == "audio_flamingo":
            self.max_frames = 30 * 16000
        else:
            self.max_frames = 120 * 16000
        self.audio_chunk = args.audio_chunk * 16000 # We set a longer audio chunk for Zipformer
        self.split_audio = args.split_audio
        self.shuffle_mc_options = bool(getattr(args, "shuffle_mc_options", False))
        self.audio_caption_style = getattr(args, "audio_caption_style", "long")
        self.broken_sample_max_retries = max(1, int(getattr(args, "broken_sample_max_retries", 32)))
        self._broken_audio_paths = set()
        
        # if true, we don't compute loss on <think>\n\n</think>\n\n
        self.skip_thinking_loss = bool(getattr(args, "skip_thinking_loss", False))

        self.data = json.load(open(args.data_path, "r"))["data"]
        self.encoder_type = encoder_type
        self.llm_type = llm_type
        self.petrel_client = Client(PETREL_CFG_PATH)

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
        lines.append('Please output your final answer strictly in JSON format. Do not include any other text, explanations, or formatting.')
        lines.append('Example: {"answer": "C"}')

        if answer_label is None:
            raise ValueError("Cannot determine correct answer for multiple-choice sample.")

        return [
            {
                "role": "user",
                "content": "\n".join(lines),
            },
            {
                "role": "assistant",
                "content": f'{{"answer": "{answer_label}"}}',
            },
        ]
        
    def _build_caption_data(self, sample):
        # we build the caption training data in the following way:
        long_caption = sample.get("long_caption", None)
        short_caption = sample.get("short_caption", None)
        
        # if both of them are None, we use the original message. This happens to the old data
        if long_caption is None and short_caption is None:
            return sample["messages"]
        
        if self.audio_caption_style == "long":
            # In this mode, we only use the long caption for training
            assert long_caption is not None, f"Sample is missing long_caption: {sample}"
            assert isinstance(long_caption, list), f"long_caption should be a list in sample: {sample}"
            
            long_caption = [c.strip() for c in long_caption if isinstance(c, str) and c.strip()]
            selected_caption = random.choice(long_caption)
            chat = sample["messages"]
            chat[1]["content"] = selected_caption
            return chat
        elif self.audio_caption_stype == "short":
            # In this mode, we only use the short caption for training
            assert short_caption is not None, f"Sample is missing short_caption: {sample}"
            assert isinstance(short_caption, list), f"short_caption should be a list in sample: {sample}"
            
            short_caption = [c.strip() for c in short_caption if isinstance(c, str) and c.strip()]
            selected_caption = random.choice(short_caption)
            chat = sample["messages"]
            chat[1]["content"] = selected_caption
            return chat
        elif self.audio_caption_style == "both":
            # in this case, we randomly choose to use either short_caption or long_caption, and sample a prompt accordingly
            assert (long_caption is not None) and (short_caption is not None), f"Sample must have both long_caption and short_caption for 'both' audio_caption_style, but got: {sample}" 
            use_short_caption = random.random() < 0.5
            caption_key = "short_caption" if use_short_caption else "long_caption"
            prompt_pool = short_captions if use_short_caption else long_captions

            caption_candidates = sample.get(caption_key, [])
            assert caption_candidates is not None, f"Sample is missing {caption_key}: {sample}"
            assert isinstance(caption_candidates, list), f"{caption_key} should be a list in sample: {sample}"

            caption_candidates = [
                c.strip() for c in caption_candidates if isinstance(c, str) and c.strip()
            ]

            selected_caption = random.choice(caption_candidates)
            selected_prompt = random.choice(prompt_pool)

            return [
                {"role": "user", "content": f"<audio>{selected_prompt}"},
                {"role": "assistant", "content": selected_caption},
            ]
    
    def _verify_chat(self, chats: List[Dict]):
        assert len(chats) == 2, f"Chat should have exactly 2 messages, but got {len(chats)}: {chats}"
        assert "<audio>" in chats[0]["content"], f"The first message should contain the audio placeholder, but got: {chats[0]}"
    
    def _add_punctuation(self, chat: List[Dict]):
        for i, message in enumerate(chat):
            content = message["content"].rstrip()
            # TODO: Deal with "," 
            if content and not content.endswith((".", "?", "!", ",")):
                chat[i]["content"] = content + "."
                # print(f"Added punctuation to message: {content} -> {chat[i]['content']}")
                
        return chat

    def __getitem__(self, index):
        for attempt in range(self.broken_sample_max_retries):
            sample_index = index if attempt == 0 else random.randint(0, len(self.data) - 1)
            sample = self.data[sample_index]
            fbanks = []
            fbank_lens = []
            raw_wavs = []
            audio_nums = []
            audio_paths = []
            has_broken_audio = False

            for audio_path in sample["audios"]:
                audio_paths.append(audio_path)
                try:
                    if audio_path.startswith("s3://"):
                        bytes_data = self.petrel_client.get(audio_path)
                        audio, fs = torchaudio.load(io.BytesIO(bytes_data))
                    else:
                        audio, fs = torchaudio.load(audio_path)
                except Exception as e:
                    if audio_path not in self._broken_audio_paths:
                        print(f"[WARN] Broken audio detected and skipped: {audio_path}; error={repr(e)}")
                        self._broken_audio_paths.add(audio_path)
                    has_broken_audio = True
                    break

                # some of our audio is not 16k hz, so we first resample them and then truncate it
                if fs != 16000:
                    audio = torchaudio.functional.resample(audio, fs, 16000)
                    fs = 16000
                audio = audio[:, :self.max_frames]

                if self.encoder_type == "zipformer2":
                    # only split the audio if longer than the audio chunk
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
                        fbank_lens[-1] = max(math.ceil((self.audio_chunk - pad_len) / 160), 50)
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

            if has_broken_audio:
                continue
            
            if sample.get("task_type") == "qa_mc" and "mc_options" in sample and "mc_question" in sample:
                chats = self._build_mc_messages(sample)
            elif sample.get("task_type") == "audio_caption":
                chats = self._build_caption_data(sample)
            else:
                chats = sample["messages"]
            
            # sometimes, the response does not end with a proper punctuation
            chats = self._add_punctuation(chats)
            self._verify_chat(chats)
            
            text = self.tokenizer.apply_chat_template(chats, tokenize=False)
            audio_token_num = (np.array(fbank_lens) // 10).sum()

            if self.llm_type == "Qwen":
                # if self.split_audio and len(audio_nums) > 0:
                #     for audio_num in audio_nums:
                #         text = text.replace("<audio>","<|vision_start|>"*audio_num+"<|vision_end|>",1)
                #     model_inputs = self.tokenizer(text, return_tensors="pt")
                # else:
                #     model_inputs = self.tokenizer(text.replace("<audio>","<|vision_start|><|vision_end|>"), return_tensors="pt")

                expanded_text = text.replace("<audio>", "<|vision_start|>" + "<|vision_pad|>" * audio_token_num + "<|vision_end|>")
                model_inputs = self.tokenizer(expanded_text, return_tensors="pt")
            elif self.llm_type == "Llama":
                model_inputs = self.tokenizer(text.replace("<audio>","<|reserved_special_token_0|><|reserved_special_token_1|>"), return_tensors="pt", add_special_tokens=False)
            else:
                raise ValueError(f"Unsupported llm_type: {self.llm_type}")
            
            input_ids = model_inputs["input_ids"][0]
            # if input_ids.shape[0] > 82:
            #     print("Encountered a long input text, which may cause OOM. Dumping the text and corresponding audio paths for debugging.")
            #     print(f"Long text sample: {sample}")
            #     print(f"Long text chat: {chats}")
            #     print(f"Long text input ids: {input_ids}")
            attention_mask = model_inputs["attention_mask"][0]

            labels = torch.full((input_ids.shape[-1],), fill_value=-100, dtype=torch.long)
            if self.llm_type == "Qwen":
                im_start_id = 151644
                im_end_id = 151645
                assistant_id = 77091
                shift_num = 3
                if self.skip_thinking_loss:
                    shift_num += 4 # the length of <think>\n\n</think>\n\n if 4
            elif self.llm_type == "Llama":
                im_start_id = 128006
                im_end_id = 128009
                assistant_id = 78191
                shift_num = 4
            else:
                raise ValueError(f"Unsupported llm_type: {self.llm_type}")
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
            new_sample["audio_paths"] = audio_paths
            new_sample["texts"] = text

            return new_sample

        raise RuntimeError(
            f"Failed to fetch a valid sample after {self.broken_sample_max_retries} retries due to broken audio files."
        )