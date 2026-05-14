#!/bin/bash
# VHA Training Launch Script
# Handles distributed training with PaddleFleet.
#
# Usage:
#   bash scripts/train.sh <config.json> [extra_args...]
#
# Examples:
#   bash scripts/train.sh config/qwen3/qwen3_gqa_1p7B_debug.json
#   bash scripts/train.sh config/qwen3/qwen3_vha_1p7B_pretrain.json --max_steps 5000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Unset platform-preset env vars (framework uses compatible upgrade)
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ID

# Cluster stability (uncomment as needed)
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_GID_INDEX=3
# export NCCL_NVLS_ENABLE=0
# export NCCL_IB_ADAPTIVE_ROUTING=1
# export FLAGS_tcp_max_syn_backlog=16384
# export CUDA_DEVICE_MAX_CONNECTIONS=1
# export FLAGS_call_stack_level=2

CONFIG=${1:?Usage: bash scripts/train.sh <config.json> [extra_args...]}
shift 1

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

EXP_NAME=$(basename "$CONFIG" .json)

# Determine launch parameters
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-36677}
NNODES=${NNODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# Auto-detect single-card debug configs
if echo "$CONFIG" | grep -q "debug"; then
    GPUS_PER_NODE=1
fi

# Kill previous training processes
bash scripts/kill_process.sh 2>/dev/null || true
sleep 1

LOG_DIR="output/${EXP_NAME}/logs"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "  VHA Training: $EXP_NAME"
echo "  Config:    $CONFIG"
echo "  Nodes:     $NNODES"
echo "  GPUs/Node: $GPUS_PER_NODE"
echo "  Master:    $MASTER_ADDR:$MASTER_PORT"
echo "  Log:       $LOG_DIR"
echo "============================================"

python -m paddle.distributed.launch \
    --master "$MASTER_ADDR:$MASTER_PORT" \
    --nnodes "$NNODES" \
    --nproc_per_node "$GPUS_PER_NODE" \
    --log_dir "$LOG_DIR" \
    --run_mode=collective \
    run_pretrain.py "$CONFIG" "$@"
