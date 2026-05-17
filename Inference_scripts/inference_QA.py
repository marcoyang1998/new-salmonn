import json
import math
import re
import warnings
import argparse

from dataclasses import dataclass, field, asdict
from modeling_salmonn import SALMONN
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
import torch.nn.functional as F
from lhotse import Fbank, FbankConfig
from transformers import AutoFeatureExtractor
import os
import soundfile as sf

# class ModelArguments:
#     model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/test_stage1_bs256_step20000_30s_whisper_beats_seperate_layernorm/checkpoint-10000"
#     base_llm_path: str = ""
#     attn_implementation: str = "flash_attention_2"
#     lora: bool = True
#     lora_rank: int = 64
#     lora_alpha: int = 64
#     lora_dropout: float = 0.05
#     llm_type: str = "Qwen"
#     encoder_type: str = "whisper_beats"
#     audio_encoder_path: str = "/mnt/bn/audio-visual-llm-data6/ckpts/beats/BEATs_iter3_plus_AS2M.pt"
#     speech_encoder_path: str = "/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2"
#     freeze_encoder: bool = True
#     connector_type: str = "MLP"
#     connector_seg_size: int = 5
#     connector_hid_size: int = 6400
    
class ModelArguments:
    # model_name_or_path: str = "output/test_stage2_step30000_120s_with_MS_spear_xlarge_token_mix_bf16/checkpoint-30000"
    # model_name_or_path: str = "output/test_stage2_step30000_120s_MC_data_only_json_format_v1_spear_xlarge_token_mix_bf16/checkpoint-30000"
    # model_name_or_path: str = "output/test_stage2_bs192_step30000_v1_data_GeminiQA_120s_spear_xlarge_token_mix_bf16/checkpoint-30000"
    # model_name_or_path: str = "output/test_stage2_bs192_step30000_v1_data_with_MC_one_letter_120s_spear_xlarge_token_mix_bf16/checkpoint-10000"
    model_name_or_path: str = "output/test_stage2_bs192_step30000_v1_data_120s_spear_xlarge_token_mix_bf16_reproduce/checkpoint-30000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    dora: bool = False
    llm_type: str = "Qwen"
    encoder_type: str = "zipformer2"
    audio_encoder_path: str = ""
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 4096
    weighted_sum_encoder: bool = False
    concat_encoder_features: bool = False
    zipformer_version: str = "xlarge"
    split_audio: bool = True
    audio_chunk: int = 120


def parse_args():
    parser = argparse.ArgumentParser(description="Run QA inference with SALMONN")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="output/test_stage2_bs192_step30000_v1_data_120s_spear_xlarge_token_mix_bf16_reproduce/checkpoint-30000",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--concat_encoder_features",
        action="store_true",
    )
    return parser.parse_args()

model_args = ModelArguments()
args = parse_args()
model_args.model_name_or_path = args.model_name_or_path
model_args.concat_encoder_features = args.concat_encoder_features
if model_args.llm_type == "Qwen":
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
elif model_args.llm_type == "Llama":
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct")
model = SALMONN.from_pretrained(
    model_args.model_name_or_path,
    config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path,"config.json")),
    model_args=model_args,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
if model_args.encoder_type == "zipformer2":
    fbank = Fbank(FbankConfig(num_mel_bins=128))
elif model_args.encoder_type == "dasheng":
    fbank = AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng", trust_remote_code=True)
elif model_args.encoder_type == "whisper_beats":
    fbank = AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2")
elif model_args.encoder_type == "qwenomni":
    fbank = AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B")
elif model_args.encoder_type == "qwen3omni":
    fbank = AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-Omni-30B-A3B-Instruct")
elif model_args.encoder_type == "mimo":
    from torchaudio.transforms import MelSpectrogram
    fbank = MelSpectrogram(sample_rate=24000, n_fft=960, hop_length=240, win_length=960, power=1.0, center=True)

json_file = "salmonn_data_v1.1/test/GeminiQA_MC_val_one_letter.json"

with open(json_file, "r") as f:
    data = json.load(f)["annotation"]

single_letter_pattern = re.compile(r"^[A-Za-z]$")
option_letter_pattern = re.compile(r"^Option\s+([A-Za-z])$", re.IGNORECASE)
total_questions = len(data)
correct_answers = 0
wrong_format_answers = 0
audio_chunk = model_args.audio_chunk * 16000

for i, item in enumerate(data):
    prompt = item["Q"]
    audio_path = item["path"]

    # prompt = """
    # Listen to the audio and answer the following multiple-choice question.\n
    # Question: By synthesizing the speaker's measured, narrative vocal delivery with the acoustic characteristics of the recording (specifically the lack of reverberation, absence of background noise, and microphone proximity), what is the most likely professional context and purpose of this audio?\n
    # Choices:\nOption A: A live radio broadcast where a host is reading a literary excerpt to a mass audience.\nOption B: A professional audiobook recording session for a work of classic fiction.\nOption C: A university professor recording a lecture on character analysis for an online literature course.\nOption D: An actor's audition tape for a period drama, recorded in a home studio.\nOption E: A voice-over narration for a historical documentary film.\n\n
    # Please output your final answer with a single letter. For example, if you think the answer is Option B, please just output 'B'.
    # """

    audio_paths = [audio_path]
    audio_num = len(audio_paths)

    messages = [
        {"role": "user", "content": "<audio>"*audio_num+prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # info = torchaudio.info(audio_paths[0])
    # audio_duration =  info.num_frames / info.sample_rate
    # if audio_duration > 120:
    #     print(f"Audio {audio_paths[0]} duration {audio_duration:.2f}s exceeds model's maximum input length. Skipping this question.")
    #     continue
    
    feature = []
    raw_wavs = []
    audio_nums = []
    split_feature_lens = []
    for audio_path in audio_paths:
        audio, fs = torchaudio.load(audio_path)
        if fs != 16000:
            audio = torchaudio.functional.resample(audio, fs, 16000)
            fs = 16000
        if model_args.encoder_type == "zipformer2":
            if model_args.split_audio and audio.shape[-1] > audio_chunk:
                if audio.size(0) > 1:
                    audio = audio.mean(dim=0, keepdim=True)
                pad_len = (-audio.shape[-1]) % audio_chunk
                if pad_len > 0:
                    audio = F.pad(audio, (0, pad_len))
                audio = audio.unfold(-1, audio_chunk, audio_chunk)[0]
                item_fbanks = fbank.extract_batch(audio, sampling_rate=fs)
                if item_fbanks.ndim == 2:
                    item_fbanks = item_fbanks.unsqueeze(0)
                audio_nums.append(item_fbanks.size(0))
                for item_fbank in item_fbanks:
                    feature.append(item_fbank)
                    split_feature_lens.append(feature[-1].size(0))
                split_feature_lens[-1] = max(math.ceil((audio_chunk - pad_len) / 160), 50)
            else:
                feature.append(fbank.extract(audio.squeeze(), sampling_rate=fs))
        elif model_args.encoder_type == "dasheng":
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            feature.append(fbank(audio, sampling_rate=fs, return_tensors="pt").input_values.squeeze(0).transpose(0, 1))
        elif model_args.encoder_type == "dasheng_wavlm":
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            feature.append(audio.squeeze())
        elif model_args.encoder_type == "whisper_beats":
            audio = audio[:,:30*16000]
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            raw_wavs.append(audio.squeeze())
            sf_audio, _ = sf.read(audio_path, frames=30*16000)
            if len(sf_audio.shape) == 2: # stereo to mono
                sf_audio = sf_audio[:, 0]
            feature.append(fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze())
        elif model_args.encoder_type == "qwenomni" or model_args.encoder_type == "qwen3omni":
            sf_audio, _ = sf.read(audio_path, frames=300*16000)
            if len(sf_audio.shape) == 2: # stereo to mono
                sf_audio = sf_audio[:, 0]
            one_fbank = fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
            feature.append(one_fbank["input_features"].squeeze())
            raw_wavs.append(one_fbank["attention_mask"].squeeze())
        elif model_args.encoder_type == "mimo":
            if audio.ndim == 2:
                audio = audio.mean(dim=0)
            audio = torchaudio.functional.resample(audio, fs, 24000)
            spec = fbank(audio[None, :])
            mel = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)
            feature.append(mel)

    if model_args.encoder_type == "whisper_beats":
        feature_lens = [f.size(0) for f in raw_wavs]
    elif model_args.encoder_type == "zipformer2" and model_args.split_audio and len(audio_nums) > 0:
        feature_lens = split_feature_lens
    else:
        feature_lens = [f.size(0) for f in feature]
    feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(model.device)
    if raw_wavs != []:
        raw_wavs = torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
    feature_lens = torch.tensor(feature_lens, device=model.device)

    if model_args.llm_type == "Qwen":
        if model_args.split_audio and len(audio_nums) > 0:
            text_for_model = text
            for one_audio_num in audio_nums:
                text_for_model = text_for_model.replace("<audio>", "<|vision_start|>" * one_audio_num + "<|vision_end|>", 1)
            model_inputs = tokenizer([text_for_model], return_tensors="pt").to(model.device)
        else:
            model_inputs = tokenizer([text.replace("<audio>","<|vision_start|><|vision_end|>")], return_tensors="pt").to(model.device)
    elif model_args.llm_type == "Llama":
        model_inputs = tokenizer([text.replace("<audio>","<|reserved_special_token_0|><|reserved_special_token_1|>")], return_tensors="pt").to(model.device)

    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        fbank_feature=feature,
        fbank_feature_len=feature_lens,
        raw_wavs=raw_wavs,
        user_prompts=[prompt],
        max_new_tokens=500
    )
    
    # parsing thinking content
    output_ids = generated_ids[0][:].tolist()
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    model_output = content.strip()
    model_output = model_output.strip(".")
    single_letter_match = single_letter_pattern.fullmatch(model_output)
    option_letter_match = option_letter_pattern.fullmatch(model_output)
    if single_letter_match is not None:
        predicted_answer = model_output.upper()
    elif option_letter_match is not None:
        predicted_answer = option_letter_match.group(1).upper()
    else:
        wrong_format_answers += 1
        warnings.warn(
            f"Invalid output format for audio {audio_path}. Expected a single letter, got: {repr(model_output)}",
            stacklevel=1,
        )
        predicted_answer = None

    ground_truth = str(item["text"]).strip().upper()
    # print(f"Predicted answer: {predicted_answer}, ground truth: {ground_truth}")
    is_correct = predicted_answer is not None and predicted_answer == ground_truth
    if is_correct:
        correct_answers += 1
        
    if i % 100 == 0 and i:
        print(f"Processed {i}/{total_questions} questions. Current accuracy: {correct_answers}/{i} ({correct_answers/i:.2%}), Wrong format answers: {wrong_format_answers}")

    # content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    # print(f"Model output: {content}, correct answer: {item['text']}")

accuracy = correct_answers / total_questions if total_questions > 0 else 0.0
checkpoint_name = os.path.basename(os.path.normpath(model_args.model_name_or_path))
print("\n===== Inference Summary =====")
print(f"0) Model checkpoint path: {model_args.model_name_or_path}")
print(f"0.1) Model checkpoint name: {checkpoint_name}")
print(f"1) Total number of questions: {total_questions}")
print(f"2) Correctly answered questions: {correct_answers} ({accuracy:.2%})")
print(f"3) Questions answered with wrong format: {wrong_format_answers}")