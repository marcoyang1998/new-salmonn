#!/bin/bash

# START=$1
# END=$2


# echo "Start = $START"
# echo "End = $END"

rlaunch \
    --memory=120000 \
    --gpu=1 \
    --cpu=12 \
    --charged-group=brainllm_gpu \
    --private-machine=yes \
    --preemptible=no \
    --mount=gpfs://gpfs1/housiyuan:/mnt/shared-storage-user/housiyuan \
    --mount=gpfs://gpfs1/brainllm-share:/mnt/shared-storage-user/brainllm-share \
    --mount=gpfs://gpfs2/brainllm2-share:/mnt/shared-storage-gpfs2/brainllm2-share \
    --mount=gpfs://gpfs2/speechllm-share:/mnt/shared-storage-gpfs2/speechllm-share \
    --custom-resources brainpp.cn/fuse=1 \
    -- bash
    

# -- bash ./run/train_96m_uniform_v2_zipformer_speech_audio_mvq_bucket_wavlm_all.sh
# --custom-resources rdma/mlnx_shared=8 \
# -e DISTRIBUTED_JOB=true \
# bash export_shar.sh --stage 18 --stop_stage 18
# > rlaunch_logs/export_yodas.log 2>&1

# -- bash
# -- bash trim_yodas.sh --start $START --end $END \
# --mount=/mnt/shared-storage-user/housiyuan/xiaoyu/workspace/icefall_general_encoder/egs/general_audio_encoder/mtl/download/gigaspeech:/mnt/shared-storage-user/housiyuan/gigaspeech \