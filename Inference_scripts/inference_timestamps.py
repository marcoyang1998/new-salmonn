import argparse
import io
import json
import math
import warnings
from tqdm import tqdm

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
from inference_utils import override_args_from_config

from petrel_client.client import Client

PETRELOSS_CONFIG = "/mnt/shared-storage-user/housiyuan/xiaoyu/petreloss.conf"
client = Client(PETRELOSS_CONFIG)

def load_audio_from_petrel_oss(audio_path: str, client: Client):
    bytes_data = client.get(audio_path)
    waveform, orig_sr = torchaudio.load(io.BytesIO(bytes_data))
    return waveform, orig_sr

def load_audio(audio: str):
    if audio.startswith("s3://"):        
        return load_audio_from_petrel_oss(audio, client)
    else:
        return torchaudio.load(audio)
    
class ModelArguments:    
    model_name_or_path: str = "output/test_all_bs192_stage2_step40000_spear_xlarge_token_mix_bf16_concat_encoder_features_True/checkpoint-30000"
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
    concat_encoder_features: bool = True
    zipformer_version: str = "xlarge"
    split_audio: bool = True
    audio_chunk: int = 120
    expand_vocab: bool = False
    inject_temporal_embedding: bool = False
    inject_temporal_embedding_nl: bool = False
    temporal_granularity: float = 1.0
    encoder_frame_rate: int = 50
    use_reasoning_network: bool = False
    reasoning_network_num_layers: int = 5
    reasoning_network_dim: int = 1024
    num_pause_steps: int = 0
    distinct_pause_embed: bool = False


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def parse_args():
    parser = argparse.ArgumentParser(description="Run QA inference with SALMONN")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="output/test_stage2_bs192_step30000_v1_data_120s_spear_xlarge_token_mix_bf16_reproduce/checkpoint-30000",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--test_json",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Path to save output JSON with model_prediction added.",
    )
    return parser.parse_args()

model_args = ModelArguments()
args = parse_args()
model_args.model_name_or_path = args.model_name_or_path

# Load latest model args from checkpoint config, then apply explicit CLI overrides.
model_args = override_args_from_config(args.model_name_or_path, model_args)

# load data
json_file = args.test_json
print(f"Loading test data from {json_file}...")
with open(json_file, "r") as f:
    json_obj = json.load(f)

if isinstance(json_obj, dict) and "data" in json_obj:
    data = json_obj["data"]
else:
    data = json_obj

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
if getattr(model_args, "inject_temporal_embedding", False):
    model.register_temporal_tokens(tokenizer)
if getattr(model_args, "inject_temporal_embedding_nl", False):
    model.register_nl_timestamp_tokenizer(tokenizer)
model.eval()
if model_args.encoder_type in ("zipformer2", "spear_transformer"):
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

audio_chunk = model_args.audio_chunk * 16000

for i, item in enumerate(tqdm(data)):
    # user prompt already contains the <audio>
    prompt = item["messages"][0]["content"]    

    audio_paths = item["audios"]
    audio_num = len(audio_paths)

    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    feature = []
    raw_wavs = []
    audio_nums = []
    split_feature_lens = []
    for audio_path in audio_paths:
        # audio, fs = torchaudio.load(audio_path)
        audio, fs = load_audio(audio_path)
        if fs != 16000:
            audio = torchaudio.functional.resample(audio, fs, 16000)
            fs = 16000
        if model_args.encoder_type in ("zipformer2", "spear_transformer"):
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
    # print(prompt)
    # print(audio_paths)

    if model_args.encoder_type == "whisper_beats":
        feature_lens = [f.size(0) for f in raw_wavs]
    elif model_args.encoder_type in ("zipformer2", "spear_transformer") and model_args.split_audio and len(audio_nums) > 0:
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
    
    # since we expanded the vocab, we do not skip special tokens
    if model_args.expand_vocab:
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=False).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=False).strip("\n")
        content = content.replace("<|im_end|>", "")
    else:
        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n") 

    model_output = content.strip()
    model_output = model_output.strip(".")
    item["model_prediction"] = model_output
    ground_truth = str(item["messages"][1]["content"]).strip()
    # print(f"model_output: {model_output}")
    # print(f"ground_truth: {ground_truth}")    

    # content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    # print(f"Model output: {content}, correct answer: {item['text']}")

# Save predictions into output JSON using the input JSON as base.
output_path = args.output_json
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
with open(output_path, "w") as f:
    if isinstance(json_obj, dict) and "data" in json_obj:
        json_obj["data"] = data
        json.dump(json_obj, f, indent=2, ensure_ascii=False)
    else:
        json.dump(data, f, indent=2, ensure_ascii=False)
print(f"Saved predictions to {output_path}")
