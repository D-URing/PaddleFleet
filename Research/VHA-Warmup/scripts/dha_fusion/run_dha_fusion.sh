#!/bin/bash
# DHA-style fusion training launcher (8-card single-node, Trainer-based).
#
# Usage:
#   bash run_dha_fusion.sh [config.json]
#
# Default config: ./dha_fusion_pretrain.json
#
# Trains:
#   GQA(8 KV heads) -> [DHA fusion] -> VHA(2 KV heads + postmix)
# starting from the GQA pretrain ckpt-24000, using ALM-augmented LM loss
# with a per-layer intra-group consistency constraint.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$SCRIPT_DIR/dha_fusion_pretrain.json}"

if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    exit 1
fi

# Activate same venv as warmup pipeline.
source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate

# Make VHA modeling code importable (qwen_provider lives there).
export PYTHONPATH="/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA:$PYTHONPATH"

# Clean any stale paddle dist envs.
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ID

# Parse output_dir from JSON to set up log dir.
OUTPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG'))['output_dir'])")
mkdir -p "$OUTPUT_DIR"
LOG_DIR="$OUTPUT_DIR/trainer-logs"
mkdir -p "$LOG_DIR"

# Snapshot config into output dir for reproducibility.
cp "$CONFIG" "$OUTPUT_DIR/dha_fusion_config.json"

# 8-card single-node launch.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"

echo "=========================================="
echo "DHA fusion training (Trainer-based)"
echo "  config:     $CONFIG"
echo "  output_dir: $OUTPUT_DIR"
echo "  log_dir:    $LOG_DIR"
echo "  GPUs:       $GPUS"
echo "=========================================="

python -m paddle.distributed.launch \
    --gpus="$GPUS" \
    --log_dir="$LOG_DIR" \
    "$SCRIPT_DIR/run_dha_fusion.py" \
    "$CONFIG"
