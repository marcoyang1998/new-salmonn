from dataclasses import dataclass
from lhotse import Fbank, FbankConfig
import torchaudio
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoFeatureExtractor
import soundfile as sf
import math
import json
import logging
import os


@dataclass
class ModelArguments:
    model_name_or_path: str = ""
    base_llm_path: str = ""
    llm_type: str = "Qwen"
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    dora: bool = False
    encoder_type: str = "zipformer2"
    audio_encoder_path: str = ""
    speech_encoder_path: str = ""
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 4096
    weighted_sum_encoder: bool = False
    concat_encoder_features: bool = False
    zipformer_version: str = "xlarge"
    split_audio: bool = True
    audio_chunk: int = 60
    expand_vocab: bool = False
    inject_temporal_embedding: bool = False
    inject_temporal_embedding_nl: bool = False
    temporal_granularity: float = 1.0
    encoder_frame_rate: int = 50
    append_reason_embed_behind_audio: bool = False
    use_reasoning_network: bool = False
    use_qwen3_embedding_model: bool = False
    qwen3_embedding_model_path: str = ""
    qwen3_embedding_tokenizer_path: str = ""
    qwen3_embedding_max_length: int = 2048
    freeze_qwen3_embedding_model: bool = True
    reasoning_network_num_layers: int = 5
    reasoning_network_dim: int = 1024
    num_pause_steps: int = 0
    distinct_pause_embed: bool = False
    encoder_lora: bool = False
    encoder_lora_rank: int = 8
    encoder_lora_alpha: int = 8
    encoder_lora_dropout: float = 0.05

OVERRIDE_KEYS = [
    "llm_type",
    "weighted_sum_encoder",
    "concat_encoder_features",
    "zipformer_version",
    "audio_encoder_path",
    "speech_encoder_path",
    "freeze_encoder",
    "connector_hid_size",
    "connector_seg_size",
    "connector_type",
    "expand_vocab",
    "inject_temporal_embedding",
    "inject_temporal_embedding_nl",
    "temporal_granularity",
    "encoder_frame_rate",
    "num_pause_steps",
    "distinct_pause_embed",
    "append_reason_embed_behind_audio",
    "use_reasoning_network",
    "use_qwen3_embedding_model",
    "qwen3_embedding_model_path",
    "qwen3_embedding_tokenizer_path",
    "qwen3_embedding_max_length",
    "freeze_qwen3_embedding_model",
    "reasoning_network_num_layers",
    "reasoning_network_dim",
    "lora",
    "dora",
    "encoder_type",
    "encoder_lora",
    "encoder_lora_rank",
    "encoder_lora_alpha",
    "encoder_lora_dropout",
]


def override_args_from_config(checkpoint_path: str, model_args):
    config_file = os.path.join(os.path.dirname(checkpoint_path), "config.json")
    if not os.path.exists(config_file):
        logging.warning(f"Config file {config_file} not found. Using default model args.")
        return model_args
    with open(config_file) as f:
        config = json.load(f)
    saved = config.get("model_args", {})
    for k in OVERRIDE_KEYS:
        val = saved.get(k)
        if val is not None:
            setattr(model_args, k, val)
            print(f"Setting {k} to {val} from checkpoint config.")
    return model_args


def maybe_init_qwen3_embedding_model(model, model_args):
    if (
        getattr(model_args, "use_reasoning_network", False)
        and getattr(model_args, "num_pause_steps", 0) > 0
        and getattr(model_args, "use_qwen3_embedding_model", False)
    ):
        model.init_qwen3_embedding_model()


def _apply_model_args_overrides(model_args, **kwargs):
    for key, value in kwargs.items():
        setattr(model_args, key, value)
    return model_args

audio_chunk = 30 * 16000
max_frames = 120 * 16000

def get_prompt(sample: dict) -> str:
    task = sample["task"]

    if task == "contextualised_asr":
        bias_words = sample.get("biasing_list", [])
        if isinstance(bias_words, list):
            bias_words_text = f"[{', '.join(str(w) for w in bias_words)}]"
        else:
            bias_words_text = f"[{str(bias_words)}]"
        if not bias_words_text:
            bias_words_text = "[]"
        return (
            "Recognize the speech and give me the transcription.\n"
            "Pay extra attention to the following contextual words:\n"
            "<biasing_list>\n"
            f"{bias_words_text}\n"
            "</biasing_list>."
        )
    
    if task in ["gender_QA", "QA", "MC_QA"]:
        prompt_template = "{}"
        return prompt_template.format(sample.get("Q", ""))
    elif task == "slot_filling":
        prompt_template = "According to the speech, what is the {}?"
        return prompt_template.format(sample.get("Q", ""))
    else:
        prompts = {
            "asr": "Recognize the speech and give me the transcription.",
            "asr_zh": "请将语音中的内容写下来。",
            "asr_de": "Hören Sie sich die Rede an und schreiben Sie ihren Inhalt auf.",
            "translation_ec": "Listen to the speech and translate it into Chinese.",
            "audiocaption": "Please describe the audio.",
            "audiocaption_v2": "Please write down what your hear in the audio.",
            "QA": "{}",
            "phone_recognition": "Provide the phonetic transcription for the speech.",
            "speech_query": "Please answer the question in detail.",
            "emotion_recognition": "Describe the emotion of the speaker in one word.",
            "lyrics_recognition": "Listen to the song and write down its content.",
            "audio_speech_description": "Describe the speech and the background audio",
            "speaker_verification": "Do you only hear the same person talking? Answer yes or no.",
            "fluent_speech_audio": "Describe the background audio and the speech in a fluent sentence.",
            "speech_separation": "Please write down what you hear each person says.",
            "audio_story_telling": "Based on the audio, write a story in detail. Your story should be highly related to the audio.",
            "speech_audio_query": "Please answer the speaker's question in detail based on the background sound.",
            "music_description": "Listen to this music clip and describe the music.",
            "translation_en2ja": "Listen to the speech and translate it into Japanese.",
            "translation_en2de": "Listen to the speech and translate it into German.",
            "speech_audio_coreasoning": "Use your strong reasoning skills to answer the speaker's question in detail based on the background sound.",
            "keywords": "Give me only three keywords of the text.",
            "speaker_diarization_asr": "Please recognize each speaker and transcribe their speech content."
        }
        
        base_prompt = prompts.get(task, "Please process the audio.")
        return base_prompt

def get_MC_template(sample: dict) -> str:
    question = sample.get("Q", "")
    options = sample.get("options", [])
    options_text = " ".join([f"{chr(65+i)}. {option}" for i, option in enumerate(options)])
    return f"{question} {options_text}"

def get_audio_path_list(sample: dict) -> list[str]:
    audio_path_list = [sample["path"]]
    if "expand_wav" in sample.keys():
        audio_path_list += sample["expand_wav"]  # for some reason expand_wav is a list!
    return audio_path_list

def extract_audio_features(audio_paths: list[str], fbank: Fbank, model, model_args, split_audio: bool = False):
    features = []
    raw_wavs = []
    audio_nums = []
    feature_lens = []
    for audio_path in audio_paths:
        audio, fs = torchaudio.load(audio_path)
        if fs != 16000:
            audio = torchaudio.functional.resample(audio, fs, 16000)
            fs = 16000
        audio = audio[:, :max_frames]

        if model_args.encoder_type == "zipformer2" or model_args.encoder_type == "spear_transformer":
            if split_audio:
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
                    features.append(item_fbank)
                    feature_lens.append(features[-1].size(0))
                feature_lens[-1] = max(math.ceil((audio_chunk - pad_len) / 160),50)
            else:
                extracted = fbank.extract(audio, sampling_rate=fs)
        elif model_args.encoder_type == "dasheng":
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            extracted = fbank(audio, sampling_rate=fs, return_tensors="pt").input_values.squeeze(0).transpose(0, 1)
        elif model_args.encoder_type == "dasheng_wavlm":
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            extracted = audio.squeeze()
        elif model_args.encoder_type == "whisper_beats":
            audio = audio[:,:30*16000]
            if audio.size(0) > 1:
                audio = audio.mean(dim=0, keepdim=True)
            raw_wavs.append(audio.squeeze())
            sf_audio, _ = sf.read(audio_path, frames=30*16000)
            if len(sf_audio.shape) == 2: # stereo to mono
                sf_audio = sf_audio[:, 0]
            extracted = fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze()
        elif model_args.encoder_type == "whisper":
            sf_audio, _ = sf.read(audio_path, frames=30*16000)
            if len(sf_audio.shape) == 2: # stereo to mono
                sf_audio = sf_audio[:, 0]
            extracted = fbank(sf_audio, sampling_rate=fs, return_tensors="pt")["input_features"].squeeze()
        elif model_args.encoder_type == "qwenomni" or model_args.encoder_type == "qwen3omni":
            sf_audio, _ = sf.read(audio_path, frames=300*16000)
            if len(sf_audio.shape) == 2: # stereo to mono
                sf_audio = sf_audio[:, 0]
            one_fbank = fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
            extracted = one_fbank["input_features"].squeeze()
            raw_wavs.append(one_fbank["attention_mask"].squeeze())
        elif model_args.encoder_type == "mimo":
            if audio.ndim == 2:
                audio = audio.mean(dim=0)
            audio = torchaudio.functional.resample(audio, fs, 24000)
            spec = fbank(audio[None, :])
            extracted = torch.log(torch.clip(spec, min=1e-7)).squeeze().transpose(0, 1)
        elif model_args.encoder_type == "perception_av":
            inputs = fbank(videos=None, audio=[audio_path], text=None)
            extracted = inputs['input_values'].squeeze()
            raw_wavs.append(inputs['padding_mask'].squeeze())
        elif model_args.encoder_type == "audio_flamingo":
            sf_audio, _ = sf.read(audio_path, frames=30*16000)
            if len(sf_audio.shape) == 2: # stereo to mono
                sf_audio = sf_audio[:, 0]
            inputs = fbank(sf_audio, sampling_rate=fs, return_tensors="pt", return_attention_mask=True)
            extracted = inputs["input_features"].squeeze()
            raw_wavs.append(inputs["attention_mask"].squeeze())

        if not split_audio:
            features.append(extracted)

    if model_args.encoder_type == "whisper_beats":
        feature_lens = [f.size(0) for f in raw_wavs]
    elif model_args.encoder_type == "audio_flamingo" or model_args.encoder_type == "qwenomni" or model_args.encoder_type == "qwen3omni" or model_args.encoder_type == "perception_av":
        feature_lens = [f.sum(-1) for f in raw_wavs]
    elif model_args.encoder_type == "zipformer2" and split_audio:
        pass
    else:
        feature_lens = [f.size(0) for f in features]
    features = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).to(model.device)
    feature_lens = torch.tensor(feature_lens, device=model.device)
    if raw_wavs != []:
        raw_wavs = torch.nn.utils.rnn.pad_sequence(raw_wavs, batch_first=True).to(model.device)
    
    return features, feature_lens, raw_wavs, audio_nums

def prepare_model_inputs(texts: list[str], audio_nums: list[int], tokenizer: AutoTokenizer, model) -> dict:
    if len(audio_nums):
        processed_text = []
        text_num = 0
        text = texts[text_num]
        for audio_num in audio_nums:
            if text.find("<audio>") == -1:
                processed_text.append(text)
                text_num += 1
                text = texts[text_num]
            text = text.replace("<audio>","<|vision_start|>"*audio_num+"<|vision_end|>",1)
    else:
        processed_text = [text.replace("<audio>", "<|vision_start|><|vision_end|>") for text in texts]
    return tokenizer(
        processed_text,
        return_tensors="pt",
        padding=True, padding_side="left"
    ).to(model.device)

def get_fbank(model_args) -> Fbank:
    if model_args.encoder_type == "zipformer2" or model_args.encoder_type == "spear_transformer":
        return Fbank(FbankConfig(num_mel_bins=128))
    elif model_args.encoder_type == "dasheng":
        return AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng", trust_remote_code=True)
    elif model_args.encoder_type == "whisper_beats" or model_args.encoder_type == "whisper":
        return AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2")
    elif model_args.encoder_type == "qwenomni":
        return AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B")
    elif model_args.encoder_type == "qwen3omni":
        return AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-Omni-30B-A3B-Instruct")
    elif model_args.encoder_type == "mimo":
        from torchaudio.transforms import MelSpectrogram
        return MelSpectrogram(sample_rate=24000, n_fft=960, hop_length=240, win_length=960, power=1.0, center=True)
    elif model_args.encoder_type == "perception_av":
        from core.audio_visual_encoder import PEAudioVisualTransform
        return PEAudioVisualTransform.from_config("/mnt/bn/audio-visual-llm-data6/ckpts/pe-av-large")
    elif model_args.encoder_type == "audio_flamingo":
        return AutoFeatureExtractor.from_pretrained("/mnt/bn/audio-visual-llm-data6/ckpts/audio-flamingo-3-hf")

def get_model_args(checkpoint_path: str):
    model_args = ModelArguments(model_name_or_path=checkpoint_path)

    if "dasheng_wavlm" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="dasheng_wavlm",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng",
            speech_encoder_path="/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm",
            connector_hid_size=8192,
        )

    elif "whisper_beats" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="whisper_beats",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data/tangchangli/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt",
            speech_encoder_path="/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2",
            connector_hid_size=6400,
        )
    
    elif "whisper" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="whisper",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2",
            connector_hid_size=6400,
        )

    elif "qwen3" in checkpoint_path and "omni" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="qwen3omni",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-Omni-30B-A3B-Instruct",
            connector_hid_size=8192,
        )

    elif "qwen" in checkpoint_path and "omni" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="qwenomni",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B",
            connector_hid_size=8192,
        )
    
    elif "mimo" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="mimo",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data6/ckpts/MiMo-Audio-Tokenizer",
        )
    
    elif "perception_av" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="perception_av",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data6/ckpts/pe-av-large",
        )

    elif "audio_flamingo" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="audio_flamingo",
            audio_encoder_path="/mnt/bn/audio-visual-llm-data6/ckpts/audio-flamingo-3-hf",
        )

    elif "spear_transformer" in checkpoint_path:
        return _apply_model_args_overrides(
            model_args,
            encoder_type="spear_transformer",
            concat_encoder_features=True,
        )

    return model_args