#!/bin/bash
# One-shot prep for ckpt-21000 branch:
#   1) build checkpoint-21000-fresh (full symlink view, including optimizer_state;
#      trainer_state.bin is rewritten with global_step=0 to reset step counter while
#      preserving consumed_samples for dataloader skip).
#      -> required by GQA continue_v4. Mirrors ckpt-24000-fresh layout used by v3.
#   2) merge sharded model_state -> model_state_merged in PaddleFleet fused layout
#      (qkv_proj / up_gate_proj fused). NO HuggingFace split — convert_gqa_to_vha_activation.py
#      and the VHA teacher loader both expect the fused layout.
set -euo pipefail

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
PRETRAIN=$ROOT/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain
SRC=$PRETRAIN/checkpoint-21000
FRESH=$PRETRAIN/checkpoint-21000-fresh
PY=$ROOT/venv_paddlefleet/bin/python

# --- 1. fresh symlink view (full mirror, then rewrite trainer_state.bin) ---
mkdir -p "$FRESH"
for entry in "$SRC"/*; do
    base=$(basename "$entry")
    # trainer_state.bin must be local (rewritten), not a symlink to source
    [[ "$base" == "trainer_state.bin" ]] && continue
    [[ -e "$FRESH/$base" ]] && continue
    ln -s "$entry" "$FRESH/$base"
done

$PY - <<PY
import paddle
src = "$SRC/trainer_state.bin"
dst = "$FRESH/trainer_state.bin"
s = paddle.load(src)
print("before:", repr(s))
s.global_step = 0
s.epoch = 0
paddle.save(s, dst)
print("after :", repr(paddle.load(dst)))
PY
echo "[ok] $FRESH ready (full symlink + reset trainer_state.bin)"

# --- 2. merge model_state -> model_state_merged (PaddleFleet fused layout, no HF split) ---
MERGED=$SRC/model_state_merged
if [[ ! -d "$MERGED" ]] || [[ -z "$(ls -A "$MERGED" 2>/dev/null)" ]]; then
    mkdir -p "$MERGED"
    $PY - <<PY
from paddle.distributed.flex_checkpoint.dcp.load_state_dict import merge_sharded_state_dict
merge_sharded_state_dict(
    load_path="$SRC/model_state",
    save_path="$MERGED",
    prefix="model",
    safetensor_prefix="model",
    offload=True,
)
PY
    echo "[ok] merged (Paddle fused): $MERGED"
else
    echo "[skip] $MERGED already exists"
fi
