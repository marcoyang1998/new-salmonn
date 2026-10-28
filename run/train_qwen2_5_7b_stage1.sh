#!/usr/bin/env bash

#######

work_dir=${WORK_DIR:-/scratches/mneme/xy316/workspace/new-salmonn}
cd "$work_dir"
echo "Current dir: $PWD"

source /scratches/mneme/xy316/softwares/miniconda3/etc/profile.d/conda.sh && conda activate salmonn

#########

export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH
export PYTHONWARNINGS=ignore

echo "---Environment info---"

echo "Num GPUs = ${GPU_COUNT:-1}"
echo "Master Address = ${MASTER_ADDR:-127.0.0.1}"
echo "Processes per Node = ${PROC_PER_NODE:-1}"
echo "Master Port = ${MASTER_PORT:-29500}"
echo "Node Count = ${NODE_COUNT:-1}"
echo "Node Rank = ${NODE_RANK:-0}"

GPU_COUNT=${GPU_COUNT:-1}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_COUNT=${NODE_COUNT:-1}
NODE_RANK=${NODE_RANK:-0}

BASE_LLM=${BASE_LLM:-/data/milsrg1/huggingface/cache/xy316/Qwen2.5-7B-Instruct}
# Stage 1 must start from Qwen2.5-7B-Instruct, not a Qwen3/Llama SALMONN checkpoint.
DATAPATH=${DATAPATH:-salmonn_data_v1.1/stage1/stage1_cv600k_as_caption_650k_lbx_asr_400k_xxn_timestamp_ovlp_timestamp_fma_msd_magna_caption.json}
DATASET_LIST=${DATASET_LIST:-salmonn_data_v1.1/dataset_lists/June_16_stage1_datasets.txt}

# data argument
split_audio=True
audio_chunk=60
max_audio_duration=40

# LLM config
lora=True
dora=False
lora_rank=64
lora_alpha=32
expand_vocab=False
skip_thinking_token_loss=False

# Qwen2.5-Instruct has no native Qwen3 thinking mode. reasoning_content fields
# are ignored by the dataset compatibility layer and normal content is trained.

# temporal embedding injection
inject_temporal_embedding_nl=True
temporal_granularity=2.0

# encoder argument
freeze_encoder=True
encoder_lr_ratio=0.1
concat_encoder_features=True
connector_seg_size=5

# bf16 checkpoint, concat features
AUDIO_ENCODER=${AUDIO_ENCODER:-/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/checkpoints/exp-600M-uniform-v2-zipformer-out-ds-2-lh-large-giga-xl-vox-all-audio-x3.0w2v2-mask-p-0.65-l-10-cha-mask-p-0.25-l-20-multi-mvq-wavlm-large-1.0-dasheng-0.2-only-mask-0-token-mix-prob-0.2-p-n-0.5-delay-2.0-md400/iter-500000-avg-2.pt}
EXP_NAME="test_stage1_qwen2_5_7B_bs512_chunk_${audio_chunk}s_concat_feature_${concat_encoder_features}_connector_seg_${connector_seg_size}_\
inject_temporal_embedding_nl_${inject_temporal_embedding_nl}_${temporal_granularity}s_\
stage1_June_16_stage1_max_duration_${max_audio_duration}s_lora_alpha_${lora_alpha}"

for required_path in "$BASE_LLM" "$DATAPATH" "$DATASET_LIST" "$AUDIO_ENCODER"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Required training path does not exist: $required_path" >&2
    echo "Override BASE_LLM, DATAPATH, DATASET_LIST, or AUDIO_ENCODER as appropriate." >&2
    exit 1
  fi
done

torchrun --nproc_per_node=$GPU_COUNT \
  --nnodes=${NODE_COUNT} --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
    train.py \
    --base_llm_path "$BASE_LLM" \
    --llm_type Qwen \
    --deepspeed deepspeed_config/zero2_no_offload_v2.json \
    --attn_implementation flash_attention_2 \
    --output_dir "output/${EXP_NAME}" \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --min_learning_rate 1e-5 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path "$DATAPATH" \
    --data_path_list "$DATASET_LIST" \
    --max_steps 40000 \
    --dataloader_num_workers 12 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 500 \
    --per_device_train_batch_size 12 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 10000 \
    --eval_strategy no \
    --report_to "tensorboard" \
    --run_name "$EXP_NAME" \
    --lora $lora \
    --lora_rank $lora_rank \
    --lora_alpha $lora_alpha \
    --dora $dora \
    --expand_vocab $expand_vocab \
    --skip_thinking_token_loss $skip_thinking_token_loss \
    --freeze_encoder $freeze_encoder \
    --encoder_lr_ratio $encoder_lr_ratio \
    --encoder_type "zipformer2" \
    --concat_encoder_features ${concat_encoder_features} \
    --audio_encoder_path "$AUDIO_ENCODER" \
    --connector_seg_size ${connector_seg_size} \
    --connector_hid_size 4096 \
    --from_scratch True \
    --split_audio $split_audio \
    --audio_chunk $audio_chunk \
    --max_audio_duration $max_audio_duration \
    --inject_temporal_embedding_nl $inject_temporal_embedding_nl \
    --temporal_granularity $temporal_granularity
