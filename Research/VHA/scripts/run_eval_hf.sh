#!/bin/bash
# Run lm_eval evaluation on HF-format checkpoint and collect results.
#
# Usage:
#   bash scripts/run_eval_hf.sh <hf_checkpoint_path> [output_name]
#
# Examples:
#   bash scripts/run_eval_hf.sh /path/to/ckpts/qwen3_GQA_1p7B_pretrain_hf
#   bash scripts/run_eval_hf.sh /path/to/ckpts/qwen3_GQA_1p7B_pretrain_hf gqa_1p7b
#
# Prerequisites:
#   pip install lm_eval transformers accelerate
#
# Notes:
#   - Uses HF_DATASETS_OFFLINE=1 to avoid network access (pre-download datasets first)
#   - Runs on single GPU (CUDA_VISIBLE_DEVICES=0)
#   - Results are saved to eval_out/<output_name>/

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Activate venv
VENV_DIR="/root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet"
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
fi

CKPT_PATH=${1:?Usage: bash scripts/run_eval_hf.sh <hf_checkpoint_path> [output_name]}
OUTPUT_NAME=${2:-$(basename "$CKPT_PATH")}

TASKS="arc_challenge,boolq,hellaswag,openbookqa,piqa,winogrande,sciq,social_iqa"
OUTPUT_DIR="./eval_out/${OUTPUT_NAME}"
LOG_FILE="./eval_out/${OUTPUT_NAME}/eval.log"

mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "  HF Evaluation"
echo "  Checkpoint: $CKPT_PATH"
echo "  Tasks:      $TASKS"
echo "  Output:     $OUTPUT_DIR"
echo "  Log:        $LOG_FILE"
echo "============================================"

# Verify checkpoint exists
if [ ! -f "$CKPT_PATH/config.json" ]; then
    echo "Error: config.json not found in $CKPT_PATH"
    echo "Make sure the checkpoint has been converted to HF format."
    echo "Run: python scripts/convert_paddle_to_hf.py --input <paddle_ckpt> --output <hf_ckpt>"
    exit 1
fi

# Run evaluation
export HF_DATASETS_OFFLINE=1
export HTTPS_PROXY=http://10.8.5.5:3128
export HTTP_PROXY=http://10.8.5.5:3128
CUDA_VISIBLE_DEVICES=0 lm_eval \
    --model hf \
    --model_args "pretrained=${CKPT_PATH},dtype=bfloat16" \
    --tasks "$TASKS" \
    --batch_size auto \
    --num_fewshot 0 \
    --output_path "$OUTPUT_DIR" \
    --log_samples 2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================"
echo "  Evaluation Complete"
echo "============================================"
echo ""

# Collect and display results
python3 "$SCRIPT_DIR/collect_eval_results.py" "$OUTPUT_DIR"
