#!/bin/bash
# Joint refine v6: teacher = ckpt-21000, init = cascade_v6.
# Adjusted 2026-05-27: reduce over-fitting to preserve CE gradient motivation.
#   - lr=1e-4, warmup 200, batch=4 (unchanged)
#   - 65536 samples (reduced from 131072), max 4000 steps, min 800 (reduced)
#   - alpha 1.0->0.1, beta 0.1->1.0, gamma 0.0->0.1 (unchanged)
#   - anneal_steps 2000 (reduced), plateau eps=0.005 (looser)
set -euo pipefail
export PYTHONPATH=$PYTHONPATH:/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
OUT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_joint_refine_v6
LOG=$OUT/runlog

mkdir -p $OUT
cd $ROOT/PaddleFleet/Research/VHA-Warmup/scripts

VHA_CKPT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v6
GQA_TEACHER=$ROOT/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-21000/model_state_merged
GQA_CONFIG=$ROOT/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA

python -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir $LOG \
    joint_refine.py \
    --gqa_checkpoint $GQA_TEACHER \
    --gqa_model_config $GQA_CONFIG \
    --vha_checkpoint $VHA_CKPT \
    --vha_model_config $VHA_CKPT \
    --output_path $OUT \
    --data_path $ROOT/datasets/fineweb-edu/qwen \
    --num_samples 65536 \
    --seq_length 2048 \
    --max_steps 4000 \
    --min_steps 800 \
    --lr 1e-4 \
    --warmup_steps 200 \
    --batch_size 4 \
    --grad_accum 1 \
    --kl_temperature 1.0 \
    --train_mode attn_norm \
    --alpha_start 1.0 --alpha_end 0.1 \
    --beta_start  0.1 --beta_end  1.0 \
    --gamma_start 0.0 --gamma_end 0.1 \
    --anneal_steps 2000 \
    --plateau_window 500 \
    --plateau_eps 0.005 \
    --plateau_metric kl \
    --log_interval 20 \
    --save_interval 500
