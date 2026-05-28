#!/bin/bash
# Joint logit refine on top of cascading-refined VHA checkpoint.
set -e

source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate
export PYTHONPATH=$(pwd)/../

unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT PADDLE_TRAINER_ID
unset OUTPUT_PATH SYS_OUTPUT_PATH COMBINED_OUTPUT_PATH

bash scripts/kill_process.sh 2>/dev/null || true

EXP_NAME=${EXP_NAME:-qwen3_vha_1p7B_joint_logit_v2}
LOG_DIR=output/${EXP_NAME}_log
mkdir -p $LOG_DIR

GQA_CHECKPOINT=${GQA_CHECKPOINT:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged}
GQA_MODEL_CONFIG=${GQA_MODEL_CONFIG:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA}
VHA_CHECKPOINT=${VHA_CHECKPOINT:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v4}
# NOTE: must use a config with num_attention_heads=16, vha_enable_premix=false (matches warmup arch).
# Qwen3-1.7B-VHA has 8Q + premix=true and is INCOMPATIBLE — would silently produce mis-shaped qkv_proj.
VHA_MODEL_CONFIG=${VHA_MODEL_CONFIG:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_cascade_v4}
OUTPUT_PATH=${OUTPUT_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/${EXP_NAME}}
DATA_PATH=${DATA_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/datasets/fineweb-edu/qwen}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_SAMPLES=${NUM_SAMPLES:-4096}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
REFINE_STEPS=${REFINE_STEPS:-2000}
LR=${LR:-1e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
GRAD_CLIP=${GRAD_CLIP:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA2=${ADAM_BETA2:-0.95}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-1}
KL_TEMPERATURE=${KL_TEMPERATURE:-1.0}
TRAIN_MODE=${TRAIN_MODE:-attn}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SAVE_INTERVAL=${SAVE_INTERVAL:-500}

python -m paddle.distributed.launch \
    --gpus="$GPUS" \
    --log_dir "$LOG_DIR" \
    ../VHA-Warmup/scripts/joint_logit_refine.py \
    --gqa_checkpoint "$GQA_CHECKPOINT" \
    --gqa_model_config "$GQA_MODEL_CONFIG" \
    --vha_checkpoint "$VHA_CHECKPOINT" \
    --vha_model_config "$VHA_MODEL_CONFIG" \
    --output_path "$OUTPUT_PATH" \
    --data_path "$DATA_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --seq_length "$SEQ_LENGTH" \
    --refine_steps "$REFINE_STEPS" \
    --lr "$LR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --grad_clip "$GRAD_CLIP" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta2 "$ADAM_BETA2" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --kl_temperature "$KL_TEMPERATURE" \
    --train_mode "$TRAIN_MODE" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL"
