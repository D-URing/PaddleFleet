#!/bin/bash
# VHA Experiment Orchestration
# Maps experiment names to configs and launches training.
#
# Usage:
#   bash exps/pretrain.sh <experiment_name> [extra_args...]
#
# Examples:
#   bash exps/pretrain.sh qwen3_gqa_1p7B_debug
#   bash exps/pretrain.sh qwen3_gqa_1p7B_pretrain_8gpu
#   bash exps/pretrain.sh qwen3_vha_1p7B_pretrain_8gpu
#   bash exps/pretrain.sh qwen3_vha_1p7B_warmup_8gpu
#   bash exps/pretrain.sh qwen3_vha_1p7B_pretrain_8gpu --max_steps 5000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

EXP_NAME=${1:?Usage: bash exps/pretrain.sh <experiment_name> [extra_args...]}
shift 1

# Experiment registry: name -> (config, gpus_per_node, nnodes)
case "$EXP_NAME" in
    # === Debug (single card) ===
    qwen3_gqa_1p7B_debug)
        CONFIG=config/qwen3/qwen3_gqa_1p7B_debug.json
        export GPUS_PER_NODE=1; export NNODES=1
        ;;
    qwen3_vha_1p7B_debug)
        CONFIG=config/qwen3/qwen3_vha_1p7B_debug.json
        export GPUS_PER_NODE=1; export NNODES=1
        ;;

    # === GQA Pretrain ===
    qwen3_gqa_1p7B_pretrain_8gpu)
        CONFIG=config/qwen3/qwen3_gqa_1p7B_pretrain.json
        export GPUS_PER_NODE=8; export NNODES=1
        ;;

    # === VHA Pretrain (from scratch) ===
    qwen3_vha_1p7B_pretrain_8gpu)
        CONFIG=config/qwen3/qwen3_vha_1p7B_pretrain.json
        export GPUS_PER_NODE=8; export NNODES=1
        ;;

    # === VHA Warmup (from GQA checkpoint) ===
    qwen3_vha_1p7B_warmup_8gpu)
        CONFIG=config/qwen3/qwen3_vha_1p7B_warmup.json
        export GPUS_PER_NODE=8; export NNODES=1
        ;;

    *)
        # Try as direct config path
        if [ -f "$EXP_NAME" ] || [ -f "config/qwen3/${EXP_NAME}.json" ]; then
            if [ -f "$EXP_NAME" ]; then
                CONFIG="$EXP_NAME"
            else
                CONFIG="config/qwen3/${EXP_NAME}.json"
            fi
            export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
            export NNODES=${NNODES:-1}
        else
            echo "Unknown experiment: $EXP_NAME"
            echo ""
            echo "Available experiments:"
            echo "  qwen3_gqa_1p7B_debug           (1 GPU, GQA debug)"
            echo "  qwen3_vha_1p7B_debug           (1 GPU, VHA debug)"
            echo "  qwen3_gqa_1p7B_pretrain_8gpu   (8 GPU, GQA pretrain)"
            echo "  qwen3_vha_1p7B_pretrain_8gpu   (8 GPU, VHA pretrain)"
            echo "  qwen3_vha_1p7B_warmup_8gpu     (8 GPU, VHA warmup from GQA)"
            echo ""
            echo "Or pass a config path directly:"
            echo "  bash exps/pretrain.sh config/qwen3/my_config.json"
            exit 1
        fi
        ;;
esac

echo "Launching experiment: $EXP_NAME"
echo "  Config: $CONFIG"
echo "  GPUs:   $GPUS_PER_NODE x $NNODES nodes"
echo ""

bash scripts/train.sh "$CONFIG" "$@"
