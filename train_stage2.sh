GPU_NUM=$(nvidia-smi -L | wc -l)
# GPU_NUM=1
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR=$METIS_WORKER_0_HOST
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
MASTER_PORT=${ports[0]}

DATAPATH='/mnt/bn/audio-visual-llm-data6/datasets/SALMONN_v1.1/AsrAstPrAacEmoMusicSepSvGenderQa_train.json'
EXP_NAME="test_all_bs256_step40000_100s_mimo_baseckpt10000"

export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH
export PYTHONWARNINGS=ignore
git config --global --add safe.directory /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN

torchrun \
    --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    train.py \
    --base_llm_path /mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B \
    --model_name_or_path /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/output/test_stage1_bs256_step20000_100s_mimo/checkpoint-10000 \
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
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 400 \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 10000 \
    --eval_strategy no \
    --report_to "wandb" \
    --encoder_type "mimo" \
    --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/MiMo-Audio-Tokenizer \
    --connector_hid_size 4096 \
    --run_name ${EXP_NAME} \