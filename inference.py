from dataclasses import dataclass, field, asdict
from modeling_salmonn import SALMONN
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
from transformers import AutoFeatureExtractor
import os
import soundfile as sf

class ModelArguments:
    model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/test_stage1_bs256_step20000_30s_whisper_beats_seperate_layernorm/checkpoint-10000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    llm_type: str = "Qwen"
    encoder_type: str = "whisper_beats"
    audio_encoder_path: str = "/mnt/bn/audio-visual-llm-data6/ckpts/beats/BEATs_iter3_plus_AS2M.pt"
    speech_encoder_path: str = "/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2"
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 6400

model_args = ModelArguments()
if model_args.llm_type == "Qwen":
    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B")
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

prompt = "Please describe the audio."
audio_paths = ["/mnt/bn/audio-visual-llm-data/yuwenyi/audiocaps/audiocaps/test/y1a8PntuXYw.wav"]
audio_num = len(audio_paths)

messages = [
    {"role": "user", "content": "<audio>"*audio_num+prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

feature = []
raw_wavs = []
for audio_path in audio_paths:
    audio, fs = torchaudio.load(audio_path)
    assert fs == 16000
    if model_args.encoder_type == "zipformer2":
        feature.append(fbank.extract(audio, sampling_rate=fs))
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
else:
    feature_lens = [f.size(0) for f in feature]
feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True).to(model.device)
if raw_wavs != []:
    raw_wavs = torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
feature_lens = torch.tensor(feature_lens, device=model.device)

if model_args.llm_type == "Qwen":
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
output_ids = generated_ids[0][:].tolist() 
content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
print(content)