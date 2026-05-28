#!/bin/bash
# VHA Warmup Training Launch Script — Single Node Multi-GPU
#
# Usage:
#   bash scripts/train_warmup_single_node.sh <config.json> [gpu_list]
#
# Examples:
#   bash scripts/train_warmup_single_node.sh config/qwen3/qwen3_vha_1p7B_warmup016.json
#   bash scripts/train_warmup_single_node.sh config/qwen3/qwen3_vha_1p7B_warmup016.json 0,1,2,3,4,5,6,7

set -e

source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate

export PYTHONPATH=$(pwd)/../

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ID

CONFIG=$1
GPU_LIST=${2:-0,1,2,3,4,5,6,7}

if [[ -z "$CONFIG" ]]; then
    echo "Usage: bash scripts/train_warmup_single_node.sh <config.json> [gpu_list]"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .json)
LOG_DIR=output/$EXP_NAME/trainer
mkdir -p $LOG_DIR

sh scripts/kill_process.sh

echo "=== Starting single-node warmup ==="
echo "  Config : $CONFIG"
echo "  GPUs   : $GPU_LIST"
echo "  LogDir : $LOG_DIR"

python -m paddle.distributed.launch \
    --devices $GPU_LIST \
    --log_dir $LOG_DIR \
    --run_mode=collective \
    run_warmup.py \
    "$CONFIG"
