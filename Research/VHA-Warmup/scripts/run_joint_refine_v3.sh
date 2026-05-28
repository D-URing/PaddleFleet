#!/bin/bash
# Joint refine v3: start from cascade_v5. Keep train_mode=attn_norm.
# Earlier v3-full run reached only KL=0.45 vs v2-attn_norm KL=0.18:
# MLP is shared with teacher, so opening it breaks the forward balance attn just aligned to.
# Improvements over v2: better init (cascade_v5), 2x samples, longer steps, stricter plateau, lower lr.
set -euo pipefail
export PYTHONPATH=$PYTHONPATH:/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
OUT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_joint_refine_v3
LOG=$OUT/runlog

mkdir -p $OUT
cd $ROOT/PaddleFleet/Research/VHA-Warmup/scripts

VHA_CKPT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v5

python -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir $LOG \
    joint_refine.py \
    --gqa_checkpoint $ROOT/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged \
    --gqa_model_config $ROOT/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA \
    --vha_checkpoint $VHA_CKPT \
    --vha_model_config $VHA_CKPT \
    --output_path $OUT \
    --data_path $ROOT/datasets/fineweb-edu/qwen \
    --num_samples 131072 \
    --seq_length 2048 \
    --max_steps 6000 \
    --min_steps 1200 \
    --lr 5e-5 \
    --warmup_steps 300 \
    --batch_size 4 \
    --grad_accum 1 \
    --kl_temperature 1.0 \
    --train_mode full \
    --alpha_start 1.0 --alpha_end 0.1 \
    --beta_start  0.1 --beta_end  1.0 \
    --gamma_start 0.0 --gamma_end 0.1 \
    --anneal_steps 2000 \
    --plateau_window 400 \
    --plateau_eps 0.003 \
    --plateau_metric kl \
    --log_interval 20 \
    --save_interval 500
