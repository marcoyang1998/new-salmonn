from modeling_salmonn import SALMONN
from lhotse import Fbank, FbankConfig
from transformers import AutoConfig, AutoTokenizer
import torch
import torchaudio
from lhotse import Fbank, FbankConfig
from transformers import AutoFeatureExtractor
import os
import soundfile as sf
from inference_utils import ModelArguments, extract_audio_features, get_fbank, maybe_init_qwen3_embedding_model, prepare_model_inputs

model_args = ModelArguments(
    model_name_or_path="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_100s_dasheng_wavlm_baseckpt10000/checkpoint-30000",
    encoder_type="dasheng_wavlm",
    audio_encoder_path="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng",
    speech_encoder_path="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm",
    connector_hid_size=8192,
)
tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B")
model = SALMONN.from_pretrained(
    model_args.model_name_or_path,
    config=AutoConfig.from_pretrained(os.path.join(model_args.model_name_or_path,"config.json")),
    model_args=model_args,
    torch_dtype="auto",
    device_map="auto"
)
maybe_init_qwen3_embedding_model(model, model_args)

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