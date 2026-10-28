#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5 stage 1 must start from the Instruct backbone. Override these paths
# for your environment; do not point STAGE1_OUTPUT at a Qwen3/Llama checkpoint.
BASE_LLM=${BASE_LLM:-/path/to/Qwen2.5-7B-Instruct}
DATA_PATH=${DATA_PATH:-/path/to/stage1.json}
AUDIO_ENCODER=${AUDIO_ENCODER:-/path/to/audio_encoder.pt}
STAGE1_OUTPUT=${STAGE1_OUTPUT:-output/qwen2_5_7b_stage1}

export PYTHONPATH=${PYTHONPATH:-$(pwd)}

torchrun --nproc_per_node="${GPU_COUNT:-1}" train.py \
  --base_llm_path "$BASE_LLM" \
  --model_name_or_path "" \
  --from_scratch True \
  --llm_type Qwen \
  --output_dir "$STAGE1_OUTPUT" \
  --data_path "$DATA_PATH" \
  --audio_encoder_path "$AUDIO_ENCODER" \
  --encoder_type zipformer2 \
  --freeze_encoder True \
  --connector_type MLP \
  --connector_seg_size 5 \
  --connector_hid_size 4096 \
  --lora True \
  --lora_rank 64 \
  --lora_alpha 64 \
  --expand_vocab False \
  --skip_thinking_token_loss False \
  --split_audio False \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 2e-4 \
  --max_steps 40000 \
  --save_steps 5000 \
  --logging_steps 10 \
  --report_to tensorboard \
  --run_name qwen2_5_7b_stage1
