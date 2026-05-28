#!/bin/bash
# Joint refine v5: v2 recipe + more data/steps. Keep lr=1e-4 (v4 lesson: lr=5e-5 too small).
# v2: lr=1e-4, 65536 samples, max 5000, plateau default -> KL=0.179 @ 2968 (then plateau)
# v5: lr=1e-4, 131072 samples, max 8000, anneal 3000, plateau eps 0.003 window 500
#   - keep v2's lr (proven sweet spot)
#   - 2x data so KL can keep dropping past v2's plateau
#   - slower anneal: block-MSE protection lasts longer, smoother handover
#   - stricter plateau: don't bail at 0.179 if more room exists
set -euo pipefail
export PYTHONPATH=$PYTHONPATH:/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
OUT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_joint_refine_v5
LOG=$OUT/runlog

mkdir -p $OUT
cd $ROOT/PaddleFleet/Research/VHA-Warmup/scripts

VHA_CKPT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v4

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
    --max_steps 8000 \
    --min_steps 1500 \
    --lr 1e-4 \
    --warmup_steps 200 \
    --batch_size 4 \
    --grad_accum 1 \
    --kl_temperature 1.0 \
    --train_mode attn_norm \
    --alpha_start 1.0 --alpha_end 0.1 \
    --beta_start  0.1 --beta_end  1.0 \
    --gamma_start 0.0 --gamma_end 0.1 \
    --anneal_steps 3000 \
    --plateau_window 500 \
    --plateau_eps 0.003 \
    --plateau_metric kl \
    --log_interval 20 \
    --save_interval 500
