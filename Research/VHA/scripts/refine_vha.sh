#!/bin/bash
# VHA cascading refinement launch script for single-node multi-GPU.
#
# Usage:
#   bash scripts/refine_vha.sh
#
# Optional overrides:
#   GPUS=1,2,3,4,5,6,7 NUM_SAMPLES=8192 BATCH_SIZE=4 GRAD_ACCUM=4 \
#   bash scripts/refine_vha.sh

source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate

export PYTHONPATH=$(pwd)/../

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ID

sh scripts/kill_process.sh

EXP_NAME=${EXP_NAME:-qwen3_vha_1p7B_refine}
LOG_DIR=output/$EXP_NAME
mkdir -p $LOG_DIR

GQA_CHECKPOINT=${GQA_CHECKPOINT:-./output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged}
GQA_MODEL_CONFIG=${GQA_MODEL_CONFIG:-./config/qwen3/Qwen3-1.7B-GQA}
VHA_CHECKPOINT=${VHA_CHECKPOINT:-./output/qwen3_vha_1p7B_init_svd_v3}
VHA_MODEL_CONFIG=${VHA_MODEL_CONFIG:-./config/qwen3/Qwen3-1.7B-VHA}
OUTPUT_PATH=${OUTPUT_PATH:-./output/qwen3_vha_1p7B_cascading_v5}
DATA_PATH=${DATA_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/datasets/fineweb-edu/qwen}

GPUS=${GPUS:-1,2,3,4,5,6,7}
NUM_SAMPLES=${NUM_SAMPLES:-1024}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
REFINE_STEPS=${REFINE_STEPS:-600}
LR=${LR:-2e-4}
LR_DECAY_RATE=${LR_DECAY_RATE:-0.90}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
GRAD_CLIP=${GRAD_CLIP:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
PATIENCE=${PATIENCE:-120}
TARGET_REDUCTION=${TARGET_REDUCTION:-0.90}
MIN_STEPS=${MIN_STEPS:-200}
GRAD_ACCUM=${GRAD_ACCUM:-1}
ADAM_BETA1_DECAY=${ADAM_BETA1_DECAY:-1.0}
ADAM_BETA2=${ADAM_BETA2:-0.95}
BATCH_SIZE=${BATCH_SIZE:-4}
DEBUG_TRACE=${DEBUG_TRACE:-0}
TRACE_INTERVAL=${TRACE_INTERVAL:-10}
TRAIN_WINDOW=${TRAIN_WINDOW:-1}
FULL_LAYER_FROM=${FULL_LAYER_FROM:-999999}
EXTRA_ARGS=""
if [[ "$DEBUG_TRACE" == "1" ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --debug_trace --trace_interval $TRACE_INTERVAL"
fi

python -m paddle.distributed.launch \
    --gpus="$GPUS" \
    --log_dir "$LOG_DIR" \
    ../VHA-Warmup/scripts/refine_vha_cascading.py \
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
    --lr_decay_rate "$LR_DECAY_RATE" \
    --warmup_ratio "$WARMUP_RATIO" \
    --grad_clip "$GRAD_CLIP" \
    --weight_decay "$WEIGHT_DECAY" \
    --patience "$PATIENCE" \
    --target_reduction "$TARGET_REDUCTION" \
    --min_steps "$MIN_STEPS" \
    --grad_accum "$GRAD_ACCUM" \
    --adam_beta1_decay "$ADAM_BETA1_DECAY" \
    --adam_beta2 "$ADAM_BETA2" \
    --batch_size "$BATCH_SIZE" \
    --train_window "$TRAIN_WINDOW" \
    --full_layer_from "$FULL_LAYER_FROM" \
    $EXTRA_ARGS
