#!/usr/bin/env bash

export PYTHONPATH=$(pwd)
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LIBRARY_PATH
export PYTHONWARNINGS=ignore
git config --global --add safe.directory /mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN_wenyi


# ckpt=output/test_all_bs192_stage2_step40000_spear_xlarge_supervised_audio_mvq_token_mix_bf16_concat_encoder_features_True/checkpoint-30000
# ckpt=/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/erha
# ckpt=output/test_stage1.5_lr_5e-5_unfreeze_encoder_lr_scale_0.2_bs256_chunk_60s_concat_feature_True/checkpoint-5000
# ckpt=/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN/output/test_stage1_bs256_step20000_120s_spear_xlarge_token_mix_bf16_concat_encoder_features_True/checkpoint-20000
ckpt=output/test_all_qwen3_8b_bs192_stage2_step40000_spear_xlarge_token_mix_bf16_concat_encoder_features_True_Qformer/checkpoint-40000

# model_id=spear_xlarge_token_mix_bf16_original_salmonn_stage2_data_30k
# model_id=spear_xlarge_token_mix_bf16_original_salmonn_stage2_data_concat_feat_40k
# model_id=spear_xlarge_supervised_audio_mvq_token_mix_bf16_original_salmonn_stage2_data_concat_feat_30k
# model_id=spear_xlarge_stage1.5_lr_5e-5_unfreeze_encoder_lr_scale_0.2_bs256_chunk_60s_concat_feature_True_5k
# model_id=spear_xlarge_stage1_token_mix_bf16_concat_encoder_features_True_20k
# model_id=spear_xlarge_stage2_token_mix_bf16_concat_encoder_features_True_connector_seg_10_40k
model_id=spear_xlarge_stage2_token_mix_bf16_concat_encoder_features_True_connector_seg_5_40k_Qformer_rerun


tasks=(
    # audiocaption
    # audiocaption_v2
    # music_description
    # emotion_recognition
    # speech_separation
    # speaker_verification
    # phone_recognition
    # asr
    # MC_QA
)

echo "CKPT: $ckpt"

# test_json="salmonn_data_v1.1/test/test_AsrAacAstPhoneSvGenderEmoSpeMusic_test_processed.json"
# test_json="salmonn_data_v1.1/test/AC_Clothov2_LSclean-other_Giga_test_processed.json"

result_dir=/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN_wenyi/results
mkdir -p $result_dir

# for task in "${tasks[@]}"; do
#     test_json="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN/salmonn_data_v1.1/test/test_AsrAacAstPhoneSvGenderEmoSpeMusic_test_processed_all.json"
#     if [[ $task == "audiocaption" ]] || [[ $task == "audiocaption_v2" ]]; then
#         test_json="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN/salmonn_data_v1.1/test/AC_Clothov2_LSclean-other_Giga_test_processed.json"
#     else
#         test_json="/mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONN/salmonn_data_v1.1/test/test_AsrAacAstPhoneSvGenderEmoSpeMusic_test_processed_all.json"
#     fi

#     python Inference_scripts/inference_batch_test_set.py \
#         --test_set_path $test_json \
#         --batch_size 16 \
#         --checkpoint_path $ckpt \
#         --write_path_title stage2_${model_id}_${task} \
#         --tasks_to_infer $task
# done

for task in "${tasks[@]}"; do
    json_file=${result_dir}/stage2_${model_id}_${task}.jsonl
    echo "Evaluating ${json_file}..."
    python /mnt/shared-storage-gpfs2/brainllm2-share/xiaoyu/SALMONNv1.1_eval/run_evaluation.py \
        --data $json_file \
        --task $task
done