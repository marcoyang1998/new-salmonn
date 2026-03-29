GPU_NUM=$(nvidia-smi -L | wc -l)
# GPU_NUM=1
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR=$METIS_WORKER_0_HOST
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
MASTER_PORT=${ports[0]}

DATAPATH='/mnt/bn/audio-visual-llm-data6/datasets/SALMONN_v1.1/Giga1000h_LS960h_lt180WavCaps_AudioCaps_Clotho_train.json'
EXP_NAME="test_stage1_bs256_step20000_120s_zipformer2_split_audio"

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
    --llm_type Qwen \
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
    --max_steps 20000 \
    --dataloader_num_workers 16 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --warmup_steps 200 \
    --per_device_train_batch_size 16 \
    --seed 42 \
    --logging_steps 10 \
    --gradient_checkpointing True \
    --gradient_accumulation_steps 2 \
    --save_strategy steps \
    --save_steps 10 \
    --eval_strategy no \
    --report_to "wandb" \
    --run_name ${EXP_NAME} \
    --encoder_type "zipformer2" \
    --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/spear-encoder-streaming-600M-speech-only/spear-xlarge-non-streaming/iter-448000-avg-2.pt \
    --connector_hid_size 4096 \
    --from_scratch True \
    --split_audio True \

# --bf16 True \

# --base_llm_path /mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-8B \
# --llm_type Qwen \

# --base_llm_path /mnt/bn/audio-visual-llm-data5/wangsiyin/models/Llama-3.1-8B-Instruct \
# --llm_type Llama \

# --encoder_type "zipformer2" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/spear-encoder-streaming-600M-speech-only/spear-xlarge-non-streaming/iter-448000-avg-2.pt \
# --connector_hid_size 4096 \

# --encoder_type "dasheng_wavlm" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/dasheng \
# --speech_encoder_path /mnt/bn/audio-visual-llm-data6/wangsiyin/SALMONN/wavlm \
# --connector_hid_size 8192 \

# --encoder_type "whisper_beats" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data/tangchangli/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
# --speech_encoder_path /mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2 \
# --connector_hid_size 6400 \

# --encoder_type "qwenomni" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data5/wangsiyin/models/Qwen2.5-Omni-7B \
# --connector_hid_size 8192 \

# --encoder_type "qwen3omni" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/Qwen3-Omni-30B-A3B-Instruct \
# --connector_hid_size 8192 \

# --encoder_type "whisper" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data/yuwenyi/ckpt/whisper/whisper_large_v2 \
# --connector_hid_size 6400 \

# --encoder_type "mimo" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/MiMo-Audio-Tokenizer \
# --connector_hid_size 4096 \

# --encoder_type "perception_av" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/pe-av-large \
# --connector_hid_size 4096 \

# --encoder_type "audio_flamingo" \
# --audio_encoder_path /mnt/bn/audio-visual-llm-data6/ckpts/audio-flamingo-3-hf \
# --connector_hid_size 4096 \