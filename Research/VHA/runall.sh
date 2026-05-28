#!/bin/bash
# VHA Master Experiment Runner
# Runs a sequence of experiments: train -> convert -> eval -> results.
#
# Usage:
#   bash runall.sh                    # Run training + eval on all nodes
#   bash runall.sh --eval-only        # Skip training, only convert + eval latest checkpoint
#
# Edit EXP_CONFIGS and parse_ranks() in selective_launch.py to control experiments.

set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Configuration
# ============================================================================
EVAL_VENV="/root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet"
CKPT_BASE="./ckpts"

EXP_CONFIGS=(
    #"config/qwen3/qwen3_gqa_1p7B_pretrain.json"
    "config/qwen3/qwen3_vha_4B_pretrain.json"
)

EVAL_ONLY=false
if [[ "$1" == "--eval-only" ]]; then
    EVAL_ONLY=true
fi

# ============================================================================
# Training (runs on all selected nodes via selective_launch.py)
# ============================================================================
if [[ "$EVAL_ONLY" == "false" ]]; then
    for CONFIG in "${EXP_CONFIGS[@]}"; do
        EXP_NAME=$(basename "$CONFIG" .json)
        echo "=========================================="
        echo "  Training: $EXP_NAME"
        echo "=========================================="

        mpirun bash scripts/train.sh "$EXP_NAME" "$CONFIG"

        if [ $? -ne 0 ]; then
            echo "ERROR: $EXP_NAME training failed, aborting."
            exit 1
        fi
        echo "$EXP_NAME training complete."
        echo ""
    done
fi

# ============================================================================
# Weight Conversion + Evaluation (runs on rank 0 only)
# ============================================================================
MY_RANK=${POD_INDEX:-0}
if [[ "$MY_RANK" != "0" ]] && [[ -n "$TRAINER_INSTANCES" ]]; then
    echo "Not rank 0, skipping eval. Exiting."
    exit 0
fi

# Activate eval venv (has transformers, lm_eval, accelerate)
source "$EVAL_VENV/bin/activate"

for CONFIG in "${EXP_CONFIGS[@]}"; do
    EXP_NAME=$(basename "$CONFIG" .json)
    OUTPUT_DIR="./output/$EXP_NAME"

    echo "=========================================="
    echo "  Post-training: $EXP_NAME"
    echo "=========================================="

    # Find the latest checkpoint (highest step number)
    LATEST_CKPT=""
    if [ -d "$OUTPUT_DIR" ]; then
        LATEST_CKPT=$(ls -d "$OUTPUT_DIR"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    fi

    if [ -z "$LATEST_CKPT" ]; then
        # No checkpoint subdirectory, try the output_dir itself
        if ls "$OUTPUT_DIR"/model-*.safetensors &>/dev/null 2>&1; then
            LATEST_CKPT="$OUTPUT_DIR"
        else
            echo "WARNING: No checkpoint found for $EXP_NAME in $OUTPUT_DIR, skipping."
            continue
        fi
    fi

    echo "  Checkpoint: $LATEST_CKPT"

    # --- Convert Paddle checkpoint to HF format ---
    HF_CKPT_DIR="${CKPT_BASE}/${EXP_NAME}_hf"
    echo "  Converting to HF format -> $HF_CKPT_DIR"

    python scripts/convert_paddle_to_hf.py \
        --input "$LATEST_CKPT" \
        --output "$HF_CKPT_DIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Weight conversion failed for $EXP_NAME"
        continue
    fi

    # --- Run HF evaluation ---
    echo "  Running evaluation..."
    bash scripts/run_eval_hf.sh "$HF_CKPT_DIR" "$EXP_NAME"

    echo ""
done

# ============================================================================
# Comparison (if multiple experiments)
# ============================================================================
EXP_NAMES=()
for CONFIG in "${EXP_CONFIGS[@]}"; do
    EXP_NAMES+=("$(basename "$CONFIG" .json)")
done

if [ ${#EXP_NAMES[@]} -gt 1 ]; then
    echo "=========================================="
    echo "  Comparison"
    echo "=========================================="
    EVAL_DIRS=()
    for NAME in "${EXP_NAMES[@]}"; do
        EVAL_DIRS+=("./eval_out/$NAME")
    done
    python3 scripts/collect_eval_results.py "${EVAL_DIRS[@]}"
fi

echo ""
echo "All done. Results in eval_out/, HF checkpoints in $CKPT_BASE/"
