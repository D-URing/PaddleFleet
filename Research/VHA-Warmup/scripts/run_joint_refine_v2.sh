#!/bin/bash
# Joint refine v2: multi-loss annealed (block_MSE + KL + final_hidden) on 65536 samples.
# Init from refine_cascade_v4 (cleanest pre-joint state) -> better starting point than joint_logit_v3.
set -euo pipefail
export PYTHONPATH=$PYTHONPATH:/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
OUT=$ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_joint_refine_v2
LOG=$OUT/runlog

mkdir -p $OUT
cd $ROOT/PaddleFleet/Research/VHA-Warmup/scripts

python -m paddle.distributed.launch \
    --gpus 0,1,2,3,4,5,6,7 \
    --log_dir $LOG \
    joint_refine.py \
    --gqa_checkpoint $ROOT/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged \
    --gqa_model_config $ROOT/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA \
    --vha_checkpoint $ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v4 \
    --vha_model_config $ROOT/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v4 \
    --output_path $OUT \
    --data_path $ROOT/datasets/fineweb-edu/qwen \
    --num_samples 65536 \
    --seq_length 2048 \
    --max_steps 5000 \
    --min_steps 800 \
    --lr 1e-4 \
    --warmup_steps 200 \
    --batch_size 4 \
    --grad_accum 1 \
    --kl_temperature 1.0 \
    --train_mode attn_norm \
    --alpha_start 1.0 --alpha_end 0.1 \
    --beta_start 0.1  --beta_end  1.0 \
    --gamma_start 0.0 --gamma_end 0.1 \
    --log_interval 20 \
    --save_interval 500
