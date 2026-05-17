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
from inference_utils import get_fbank, extract_audio_features, prepare_model_inputs

class ModelArguments:
    model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_100s_dasheng_wavlm_baseckpt10000/checkpoint-30000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    dora: bool = False
    encoder_type: str = "dasheng_wavlm" # zipformer2
    audio_encoder_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng" # ""
    speech_encoder_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm" # ""
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 8192 # 4096

model_args = ModelArguments()
tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B")
model = SALMONN.from_pretrained(
    model_args.model_name_or_path,
    config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path,"config.json")),
    model_args=model_args,
    torch_dtype="auto",
    device_map="auto"
)

fbank = get_fbank(model_args)

prompt = "Please give me the transcription."
audio_paths = ["/mnt/bn/audio-visual-llm-data/datasets/GigaSpeech/preprocessed_data/subset_dev/audio/POD1000000022/POD1000000022_S0000026.wav"]
audio_num = len(audio_paths)

messages = [
    {"role": "user", "content": "<audio>"*audio_num+prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

text = [text]

feature, feature_lens, raw_wavs = extract_audio_features(audio_paths, fbank, model, model_args)

model_inputs = prepare_model_inputs(text, tokenizer, model)

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