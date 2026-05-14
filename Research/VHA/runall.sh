#!/bin/bash
# VHA Master Experiment Runner
# Runs a sequence of experiments: train -> eval -> results.
#
# Usage:
#   bash runall.sh
#
# Edit EXP_NAMES below to control which experiments to run.

set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Data setup
# ============================================================================
if [ ! -e /datasets/fineweb-edu ]; then
    echo "Warning: /datasets/fineweb-edu not found."
    echo "Please create a symlink or set input_dir in your config."
fi

# ============================================================================
# Experiments to run (edit this list)
# ============================================================================
EXP_CONFIGS=(
    "config/qwen3/qwen3_gqa_1p7B_pretrain.json"
    "config/qwen3/qwen3_vha_1p7B_pretrain.json"
    # "config/qwen3/qwen3_vha_1p7B_warmup.json"
)

# ============================================================================
# Training
# ============================================================================
for CONFIG in "${EXP_CONFIGS[@]}"; do
    EXP_NAME=$(basename "$CONFIG" .json)
    echo "=========================================="
    echo "  Training: $EXP_NAME"
    echo "=========================================="

    bash scripts/train.sh "$CONFIG"

    if [ $? -ne 0 ]; then
        echo "ERROR: $EXP_NAME training failed, aborting."
        exit 1
    fi
    echo "$EXP_NAME training complete."
    echo ""
done

# ============================================================================
# Evaluation
# ============================================================================
echo "=========================================="
echo "  Running Evaluations"
echo "=========================================="

EXP_NAMES=()
for CONFIG in "${EXP_CONFIGS[@]}"; do
    EXP_NAMES+=("$(basename "$CONFIG" .json)")
done

for EXP_NAME in "${EXP_NAMES[@]}"; do
    echo "Evaluating: $EXP_NAME"
    bash scripts/eval.sh "$EXP_NAME" || echo "Warning: eval for $EXP_NAME had issues"
done

# Wait for background eval processes
echo "Waiting for evaluations to complete..."
wait

# ============================================================================
# Results
# ============================================================================
echo "=========================================="
echo "  Results"
echo "=========================================="
python scripts/get_results.py "${EXP_NAMES[@]}"

# ============================================================================
# Plot
# ============================================================================
python scripts/plot_loss.py "${EXP_NAMES[@]}" --output output/loss_comparison.png

echo ""
echo "All done. Results in eval_out/, plots in output/loss_comparison.png"
