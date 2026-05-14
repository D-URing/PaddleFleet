#!/bin/bash
# VHA Evaluation Script
# Runs lm_eval benchmarks in parallel across multiple tasks.
#
# Usage:
#   bash scripts/eval.sh <experiment_name> [checkpoint_path]
#
# Examples:
#   bash scripts/eval.sh qwen3_gqa_1p7B_pretrain
#   bash scripts/eval.sh qwen3_vha_1p7B_pretrain ./output/qwen3_vha_1p7B_pretrain/checkpoint-24000
#
# Prerequisites:
#   pip install lm_eval transformers

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Unset platform env vars for single-node eval
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINERS_NUM
unset PADDLE_TRAINER_ID
export PADDLE_TRAINERS_NUM=1
export PADDLE_TRAINER_ID=0

EXP_NAME=${1:?Usage: bash scripts/eval.sh <experiment_name> [checkpoint_path]}
CKPT_PATH=${2:-"./output/${EXP_NAME}/checkpoint-latest"}

if [ ! -d "$CKPT_PATH" ]; then
    echo "Warning: Checkpoint path not found: $CKPT_PATH"
    echo "Trying to find latest checkpoint in output/${EXP_NAME}/"
    CKPT_PATH=$(ls -d ./output/${EXP_NAME}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$CKPT_PATH" ]; then
        echo "Error: No checkpoint found for experiment: $EXP_NAME"
        exit 1
    fi
    echo "Using checkpoint: $CKPT_PATH"
fi

# Evaluation task groups (run in parallel on different GPUs)
TASKS=(
    "arc_challenge"
    "arc_easy"
    "hellaswag"
    "openbookqa"
    "boolq"
    "piqa"
    "winogrande"
    "sciq"
)

NUM_SHOT=0
OUTPUT_DIR="./eval_out/${EXP_NAME}"
LOG_DIR="./output/${EXP_NAME}/eval_logs"
mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "============================================"
echo "  VHA Evaluation: $EXP_NAME"
echo "  Checkpoint: $CKPT_PATH"
echo "  Tasks:      ${TASKS[*]}"
echo "  Output:     $OUTPUT_DIR"
echo "============================================"

NUM_GPUS=$(python -c "import paddle; print(paddle.device.cuda.device_count())" 2>/dev/null || echo "8")
NUM_TASKS=${#TASKS[@]}

for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu_id=$((i % NUM_GPUS))

    log_file="${LOG_DIR}/eval_${task}.log"

    echo "Starting eval: $task (GPU $gpu_id) -> $log_file"

    CUDA_VISIBLE_DEVICES=$gpu_id nohup lm_eval \
        --model hf \
        --model_args "pretrained=${CKPT_PATH},trust_remote_code=True,dtype=bfloat16" \
        --tasks "$task" \
        --batch_size auto \
        --output_path "${OUTPUT_DIR}" \
        --log_samples \
        --num_fewshot $NUM_SHOT > "$log_file" 2>&1 &

    echo "  PID: $!"
    sleep 0.5
done

echo ""
echo "All evaluation tasks launched in background."
echo "Monitor with: tail -f ${LOG_DIR}/eval_*.log"
echo "Aggregate results with: python scripts/get_results.py $EXP_NAME"
