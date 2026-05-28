#!/bin/bash
# PCA init from GQA ckpt-21000 (re-do PCA conversion).
# Output goes to qwen3_vha_1p7B_kv_postmix_activation_128_k21
#
# Replicates run_vha_pipeline.sh stages: collect -> convert
#
# Usage:
#   bash scripts/run_pca_init_k21.sh
set -euo pipefail
export EXP_NAME=qwen3_vha_1p7B_kv_postmix_activation_128_k21
export ACTIVATION_SAMPLES=128
export NUM_GPUS=8

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
WARMUP_DIR=$ROOT/PaddleFleet/Research/VHA-Warmup

# Override GQA_CKPT to point at ckpt-21000
sed -E 's|checkpoint-24000/model_state_merged|checkpoint-21000/model_state_merged|' \
    "$WARMUP_DIR/scripts/run_vha_pipeline.sh" > "$WARMUP_DIR/scripts/run_vha_pipeline_k21.sh"
chmod +x "$WARMUP_DIR/scripts/run_vha_pipeline_k21.sh"

bash "$WARMUP_DIR/scripts/run_vha_pipeline_k21.sh" collect
bash "$WARMUP_DIR/scripts/run_vha_pipeline_k21.sh" convert

# Final init dir lives at $WARMUP_DIR/output/$EXP_NAME/init
# But cascade/joint expect a top-level dir. Symlink:
INIT_TOP=$WARMUP_DIR/output/$EXP_NAME
if [[ ! -e "$INIT_TOP/config.json" ]]; then
    for f in "$INIT_TOP/init"/*; do
        ln -sf "$f" "$INIT_TOP/$(basename $f)"
    done
fi
echo "[ok] VHA init ready at: $INIT_TOP"
