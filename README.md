# Basic files
Model: modeling_salmonn.py

Encoder: spear_encoder/

Dataset: datasets.py

Train: train.py

# Training Locally
Stage 1: train.sh 

Stage 2: train_stage2.sh

# Training on Arnold machine
Stage 1: train_arnold.sh 

Stage 2: train_stage2_arnold.sh

# inference
Single sample: inference.py

# Available Checkpoint 
Stage 1: /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_stage1_bs256_step40000_2s/checkpoint-5000

Stage 2: /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs192_step40000_2s/checkpoint-30000

# TODO
- [ ] Performance evalution on benchmark
- [ ] Try longer audio (currently only support up to 2 minutes)
- [ ] Try new xiaoyu encoder (torchaudio fbank)
- [ ] Try new training template (available audio anywhere)
- [ ] Try new training strategy (maybe different training stages)
- [ ] Try new training data (more data; new caption)

# inference with zipformer encoder (SPEAR)

```
class ModelArguments:
    model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs192_step40000_2s/checkpoint-30000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    llm_type: str = "Qwen"
    encoder_type: str = "zipformer2"
    audio_encoder_path: str = ""
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 4096
```

# inference with dashengwavlm

```
class ModelArguments:
    model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_100s_dasheng_wavlm_baseckpt10000/checkpoint-30000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    llm_type: str = "Qwen"
    encoder_type: str = "dasheng_wavlm"
    audio_encoder_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng"
    speech_encoder_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm"
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 8192
```

# inference with whisperbeats

```
class ModelArguments:
    model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_30s_whisper_beats_baseckpt10000/checkpoint-30000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    llm_type: str = "Qwen"
    encoder_type: str = "whisper_beats"
    audio_encoder_path: str = "/mnt/bn/audio-visual-llm-data/tangchangli/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
    speech_encoder_path: str = "/mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2"
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 6400
```

# inference with qwenomni encoder

```
class ModelArguments:
    model_name_or_path: str = "/mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_100s_qwen_baseckpt10000/checkpoint-30000"
    base_llm_path: str = ""
    attn_implementation: str = "flash_attention_2"
    lora: bool = True
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    llm_type: str = "Qwen"
    encoder_type: str = "qwenomni"
    audio_encoder_path: str = "/mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B"
    speech_encoder_path: str = ""
    freeze_encoder: bool = True
    connector_type: str = "MLP"
    connector_seg_size: int = 5
    connector_hid_size: int = 8192
```

# inference code

```
python Inference_scripts/inference_batch_test_set.py --test_set_path /mnt/bn/audio-visual-llm-data3/datasets/multitask_json/AC_Clothov2_LSclean-other_Giga_test.json --tasks_to_infer audiocaption_v2 --checkpoint_path /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_30s_120s_zipformer2_split_audio_baseckpt10000/checkpoint-30000 --write_path_title zipformer2_split_audio_stage2_ckpt30000_clotho

python Inference_scripts/inference_batch_test_set.py --checkpoint_path /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_all_bs256_step40000_30s_120s_zipformer2_split_audio_baseckpt10000/checkpoint-30000 --write_path_title zipformer2_split_audio_stage2_ckpt30000_default
```


# Evaluation


```
python /mnt/bn/audio-visual-llm-data6/terumi/SALMONNv1.1_eval/run_evaluation.py --task audiocaption --data /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/results/whisper_beats_newprecision_ckpt25000_audiocaption.jsonl
```