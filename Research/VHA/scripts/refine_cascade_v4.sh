#!/bin/bash
# Cascade refine v4: corrected architecture (16Q + postmix-only).
# Starts from attn8 (the only ckpt with the right warmup-compatible arch).
set -e

source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate
export PYTHONPATH=$(pwd)/../

unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT PADDLE_TRAINER_ID
unset OUTPUT_PATH SYS_OUTPUT_PATH COMBINED_OUTPUT_PATH

bash scripts/kill_process.sh 2>/dev/null || true

EXP_NAME=${EXP_NAME:-qwen3_vha_1p7B_refine_cascade_v4}
LOG_DIR=output/${EXP_NAME}_log
mkdir -p $LOG_DIR

GQA_CHECKPOINT=${GQA_CHECKPOINT:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged}
GQA_MODEL_CONFIG=${GQA_MODEL_CONFIG:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA}
# attn8 is the only ckpt with correct 16Q + postmix-only arch matching warmup.
# Use it as both VHA_CHECKPOINT and VHA_MODEL_CONFIG to avoid arch mismatch.
VHA_CHECKPOINT=${VHA_CHECKPOINT:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_alignment_attn8}
VHA_MODEL_CONFIG=${VHA_MODEL_CONFIG:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_refine_alignment_attn8}
OUTPUT_PATH=${OUTPUT_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/${EXP_NAME}}
DATA_PATH=${DATA_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/datasets/fineweb-edu/qwen}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
NUM_SAMPLES=${NUM_SAMPLES:-4096}
SEQ_LENGTH=${SEQ_LENGTH:-2048}
REFINE_STEPS=${REFINE_STEPS:-1500}
LR=${LR:-2e-4}
LR_DECAY_RATE=${LR_DECAY_RATE:-0.92}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
GRAD_CLIP=${GRAD_CLIP:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
PATIENCE=${PATIENCE:-100}
TARGET_REDUCTION=${TARGET_REDUCTION:-0.90}
MIN_STEPS=${MIN_STEPS:-150}
GRAD_ACCUM=${GRAD_ACCUM:-1}
ADAM_BETA2=${ADAM_BETA2:-0.95}
BATCH_SIZE=${BATCH_SIZE:-2}
TRAIN_WINDOW=${TRAIN_WINDOW:-1}
TRAIN_MODE=${TRAIN_MODE:-attn}
TARGET_MODE=${TARGET_MODE:-student_input}

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
    --adam_beta2 "$ADAM_BETA2" \
    --batch_size "$BATCH_SIZE" \
    --train_window "$TRAIN_WINDOW" \
    --train_mode "$TRAIN_MODE" \
    --target_mode "$TARGET_MODE"
