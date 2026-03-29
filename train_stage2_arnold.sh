
cd /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN
git config --global --add safe.directory /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN

source /mnt/bn/audio-visual-llm-data6/yuwenyi/conda_env.sh
conda activate salmonn_cu126
export TORCH_HOME=/mnt/bn/audio-visual-llm-data/torch_home
export LD_LIBRARY_PATH=/mnt/bn/audio-visual-llm-data6/yuwenyi/miniconda3/envs/salmonn_cu126/lib
export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH
export PYTHONWARNINGS=ignore
export DECORD_REWIND_RETRY_MAX=256
set -u
export HADOOP_ROOT_LOGGER="ERROR,console"
export LIBHDFS_OPTS="-Dhadoop.root.logger=$HADOOP_ROOT_LOGGER"
export LIBHDFS_OPTS="$LIBHDFS_OPTS -Xms512m -Xmx10g "
export KRB5CCNAME="/tmp/krb5cc"
export TF_CPP_MIN_LOG_LEVEL="2"
export MKL_THREADING_LAYER="GNU"
export NCCL_IB_GID_INDEX="3"
export NCCL_IB_DISABLE="0"
export NCCL_IB_HCA="mlx5_2:1"
export NCCL_SOCKET_IFNAME="eth0"
export ARNOLD_FRAMEWORK="pytorch"
export NCCL_DEBUG=ERROR
GPU_COUNT=$(python3 -c "import torch;print(torch.cuda.device_count())")
if [ $GPU_COUNT -gt 0 ]; then
    NUM_TRAINERS=$GPU_COUNT
else
    NUM_TRAINERS=${NUM_TRAINERS:-1}
fi
NNODES=${ARNOLD_WORKER_NUM:-1}
NUM_TRAINERS=${NUM_TRAINERS:-auto}
JOB_ID=${ARNOLD_TRIAL_ID:-40001}
HOST_NODE_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
if [ $NNODES -gt 1 ]; then
  HOST_NODE_PORT=$ARNOLD_WORKER_0_PORT
else
    while true; do
        # Generate a random port number between 1024 and 65535
        PORT=$((RANDOM % (65535 - 1024 + 1) + 1024))
        # Check if the port is available using 'nc'
        nc -z 127.0.0.1 $PORT >/dev/null 2>&1
        # If the port is available, break the loop
        if [[ $? -ne 0 ]]; then
            break
        fi
    done
    HOST_NODE_PORT=$PORT
fi

DATAPATH='/mnt/bn/audio-visual-llm-data6/datasets/SALMONN_v1.1/AsrAstPrAacEmoMusicSepSvGenderQa_train.json'
EXP_NAME="arnold_all_bs256_step40000_30s_120s_zipformer2_split_audio_baseckpt10000"

torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NUM_TRAINERS \
    --rdzv_id=$JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOST_NODE_ADDR:$HOST_NODE_PORT \
    train.py \
    --base_llm_path /mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B \
    --model_name_or_path /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/arnold_stage1_bs256_step20000_120s_zipformer2_split_audio/checkpoint-10000 \
    --deepspeed /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/deepspeed_config/zero2.json \
    --attn_implementation flash_attention_2 \
    --output_dir "output/"${EXP_NAME} \
    --learning_rate 2e-4 \
    --weight_decay 0.1 \
    --min_learning_rate 1e-5 \
    --max_grad_norm 5.0 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-6 \
    --bf16 True \
    --tf32 True \
    --data_path ${DATAPATH} \
    --max_steps 40000 \
    --dataloader_num_workers 32 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 400 \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 1 \
    --save_strategy steps \
    --save_steps 10000 \
    --eval_strategy no \
    --report_to "wandb" \
    --encoder_type "zipformer2" \
    --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/spear-encoder-streaming-600M-speech-only/spear-xlarge-non-streaming/iter-448000-avg-2.pt \
    --connector_hid_size 4096 \
    --split_audio True \
    --run_name ${EXP_NAME} \

# --bf16 True \