#!/bin/bash
# Cascade refine v6: teacher = ckpt-21000, init = k21 PCA.

set -euo pipefail
export PYTHONPATH=$PYTHONPATH:/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
OUT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v6
LOG=$OUT/runlog

mkdir -p $OUT
cd $ROOT/PaddleFleet/Research/VHA-Warmup/scripts

VHA_INIT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_kv_postmix_activation_128_k21
GQA_TEACHER=$ROOT/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-21000/model_state_merged
GQA_CONFIG=$ROOT/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA

python -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir $LOG \
    refine_vha_cascading.py \
    --gqa_checkpoint $ROOT/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-21000/model_state_merged \
    --gqa_model_config $ROOT/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA \
    --vha_checkpoint $VHA_INIT \
    --vha_model_config $VHA_INIT \
    --output_path $OUT \
    --data_path $ROOT/datasets/fineweb-edu/qwen \
    --num_samples 16384 \
    --seq_length 2048 \
    --batch_size 4 \
    --grad_accum 1 \
    --refine_steps 4000 \
    --min_steps 600 \
    --patience 400 \
    --target_reduction 0.96 \
    --train_window 4 \
    --segment_stride 2 \
    --train_mode attn \
    --target_mode student_input \
    --lr 5e-4 \
    --lr_decay_rate 0.88 \
    --lr_min_ratio 0.05 \
    --warmup_ratio 0.05 \
    --grad_clip 1.0 \
    --weight_decay 0.1
