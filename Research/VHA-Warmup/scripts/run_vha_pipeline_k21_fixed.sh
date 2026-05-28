#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/paddlejob/share-storage/gpfs/system-public/dingxibo
VHA_DIR="$ROOT/PaddleFleet/Research/VHA"
WARMUP_DIR="$ROOT/PaddleFleet/Research/VHA-Warmup"
PY="$ROOT/venv_paddlefleet/bin/python"

GQA_CKPT="$VHA_DIR/output/qwen3_gqa_1p7B_pretrain/checkpoint-21000/model_state_merged"
GQA_CONFIG="$VHA_DIR/config/qwen3/Qwen3-1.7B-GQA"
DATA_PREFIX="$ROOT/datasets/fineweb-edu/qwen"
DATA_DIR="$ROOT/datasets/fineweb-edu"
TOKENIZER="$VHA_DIR/config/qwen3/tokenizer"

EXP_NAME=${EXP_NAME:-qwen3_vha_1p7B_clean_kv_postmix}
WORK_DIR="$WARMUP_DIR/output/$EXP_NAME"
CACHE_PATH="$WORK_DIR/activation_cache_128x4096_merged.npz"
INIT_DIR="$WORK_DIR/init"
REFINE_DIR="$WORK_DIR/refine"
WARMUP_CONFIG="$WORK_DIR/warmup.json"
WARMUP_OUT="$WORK_DIR/warmup"
PART_DIR="$INIT_DIR/conversion_parts"
LOG_DIR="$WORK_DIR/logs"

NUM_GPUS=${NUM_GPUS:-8}
SEQ_LENGTH=${SEQ_LENGTH:-4096}
ACTIVATION_SAMPLES=${ACTIVATION_SAMPLES:-128}
ACTIVATION_BATCH_SIZE=${ACTIVATION_BATCH_SIZE:-1}
CONVERSION_MAX_TOKENS=${CONVERSION_MAX_TOKENS:-65536}
EVAL_SAMPLES=${EVAL_SAMPLES:-1}
EVAL_SEQ_LENGTH=${EVAL_SEQ_LENGTH:-256}
REFINE_SAMPLES=${REFINE_SAMPLES:-256}
REFINE_SEQ_LENGTH=${REFINE_SEQ_LENGTH:-1024}
REFINE_STEPS=${REFINE_STEPS:-200}
REFINE_BATCH_SIZE=${REFINE_BATCH_SIZE:-2}
WARMUP_MAX_STEPS=${WARMUP_MAX_STEPS:-12000}
WARMUP_BATCH_SIZE=${WARMUP_BATCH_SIZE:-4}
WARMUP_LR=${WARMUP_LR:-2e-4}
WARMUP_DISTILL_ALPHA=${WARMUP_DISTILL_ALPHA:-0.0}
WARMUP_LAYER_BETA=${WARMUP_LAYER_BETA:-0.0}

usage() {
  cat <<EOF
Usage: bash scripts/run_vha_pipeline.sh <stage>
Stages: clean, collect, convert, eval-init, refine, eval-refine, make-warmup-config, warmup, all
EOF
}

merge_cache() {
  "$PY" - <<PY
import json, os, zipfile, numpy as np
out = "$CACHE_PATH"
shards = [os.path.join("$WORK_DIR", f"activation_shard{rank}.npz") for rank in range($NUM_GPUS)]
missing = [p for p in shards if not os.path.exists(p)]
if missing:
    raise SystemExit(f"missing activation shards: {missing}")
os.makedirs(os.path.dirname(out), exist_ok=True)
with np.load(shards[0], allow_pickle=False) as first:
    keys = [k for k in first.files if k != "__meta__"]
with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
    for key in keys:
        arrays = []
        for shard in shards:
            with np.load(shard, allow_pickle=False) as data:
                arrays.append(data[key])
        merged = np.concatenate(arrays, axis=0)
        tmp = out + f".{key}.npy"
        np.save(tmp, merged)
        zf.write(tmp, arcname=f"{key}.npy")
        os.remove(tmp)
        print(f"merged {key}: {merged.shape} {merged.dtype}", flush=True)
    meta = {"source_shards": shards, "num_shards": len(shards), "seq_length": $SEQ_LENGTH, "samples": $ACTIVATION_SAMPLES}
    tmp = out + ".__meta__.npy"
    np.save(tmp, np.array(json.dumps(meta), dtype=np.unicode_))
    zf.write(tmp, arcname="__meta__.npy")
    os.remove(tmp)
print(f"merged cache saved: {out}", flush=True)
PY
}

run_collect() {
  mkdir -p "$WORK_DIR" "$LOG_DIR"
  local per_rank=$((ACTIVATION_SAMPLES / NUM_GPUS))
  if [[ $per_rank -lt 1 ]]; then echo "ACTIVATION_SAMPLES must be >= NUM_GPUS"; exit 1; fi
  for r in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r "$PY" "$WARMUP_DIR/scripts/convert_gqa_to_vha_activation.py" \
      --gqa_checkpoint "$GQA_CKPT" --gqa_model_config "$GQA_CONFIG" --output_path "$INIT_DIR" \
      --calib_data "$DATA_DIR" --num_calib_samples "$per_rank" --seq_length "$SEQ_LENGTH" \
      --activation_cache "$WORK_DIR/activation_shard${r}.npz" --save_activation_cache --refresh_activation_cache \
      --activation_shard_rank "$r" --activation_shard_count "$NUM_GPUS" --activation_batch_size "$ACTIVATION_BATCH_SIZE" \
      --conversion_mode kv_postmix_only --collect_only > "$LOG_DIR/collect_rank${r}.log" 2>&1 &
  done
  wait
  merge_cache
  rm -f "$WORK_DIR"/activation_shard*.npz
}

run_convert() {
  mkdir -p "$INIT_DIR" "$PART_DIR" "$LOG_DIR"
  rm -f "$PART_DIR"/*.pkl
  for r in $(seq 0 $((NUM_GPUS - 1))); do
    CUDA_VISIBLE_DEVICES=$r "$PY" "$WARMUP_DIR/scripts/convert_gqa_to_vha_activation.py" \
      --gqa_checkpoint "$GQA_CKPT" --gqa_model_config "$GQA_CONFIG" --output_path "$INIT_DIR" \
      --activation_cache "$CACHE_PATH" --seq_length "$SEQ_LENGTH" --conversion_mode kv_postmix_only \
      --conversion_max_tokens "$CONVERSION_MAX_TOKENS" --conversion_layer_rank "$r" --conversion_layer_count "$NUM_GPUS" \
      --conversion_part_dir "$PART_DIR" > "$LOG_DIR/convert_rank${r}.log" 2>&1 &
  done
  wait
  "$PY" "$WARMUP_DIR/scripts/convert_gqa_to_vha_activation.py" \
    --gqa_checkpoint "$GQA_CKPT" --gqa_model_config "$GQA_CONFIG" --output_path "$INIT_DIR" \
    --activation_cache "$CACHE_PATH" --conversion_mode kv_postmix_only --assemble_from_parts "$PART_DIR" \
    > "$LOG_DIR/convert_assemble.log" 2>&1
}

run_eval() {
  local ckpt=$1
  local name=$2
  mkdir -p "$LOG_DIR"
  CUDA_VISIBLE_DEVICES=${EVAL_GPU:-0} "$PY" "$WARMUP_DIR/scripts/eval_vha_logits_loss.py" \
    --gqa_checkpoint "$GQA_CKPT" --vha_checkpoint "$ckpt" --gqa_model_config "$GQA_CONFIG" --vha_model_config "$ckpt" \
    --data_path "$DATA_PREFIX" --num_samples "$EVAL_SAMPLES" --seq_length "$EVAL_SEQ_LENGTH" --batch_size 1 \
    --output_json "$WORK_DIR/${name}_logits_loss.json" | tee "$LOG_DIR/${name}_eval.log"
}

run_refine() {
  mkdir -p "$REFINE_DIR" "$LOG_DIR"
  CUDA_VISIBLE_DEVICES=${REFINE_GPU:-0} "$PY" "$WARMUP_DIR/scripts/refine_vha_cascading.py" \
    --single_process --gqa_checkpoint "$GQA_CKPT" --gqa_model_config "$GQA_CONFIG" \
    --vha_checkpoint "$INIT_DIR" --vha_model_config "$INIT_DIR" --output_path "$REFINE_DIR" \
    --data_path "$DATA_PREFIX" --num_samples "$REFINE_SAMPLES" --seq_length "$REFINE_SEQ_LENGTH" \
    --refine_steps "$REFINE_STEPS" --batch_size "$REFINE_BATCH_SIZE" \
    --train_window 2 --segment_stride 1 --train_mode attn_norm > "$LOG_DIR/refine.log" 2>&1
}

make_warmup_config() {
  mkdir -p "$WORK_DIR"
  cat > "$WARMUP_CONFIG" <<JSON
{
  "model_name_or_path": "$REFINE_DIR",
  "tokenizer_name_or_path": "$TOKENIZER",
  "init_checkpoint": "$REFINE_DIR",
  "input_dir": "$DATA_DIR",
  "output_dir": "$WARMUP_OUT",
  "max_seq_length": 4096,
  "sharding": "stage1",
  "sharding_degree": 128,
  "tensor_model_parallel_size": 1,
  "pipeline_model_parallel_size": 1,
  "sequence_parallel": false,
  "per_device_train_batch_size": $WARMUP_BATCH_SIZE,
  "per_device_eval_batch_size": 4,
  "gradient_accumulation_steps": 1,
  "learning_rate": $WARMUP_LR,
  "min_learning_rate": 1e-5,
  "adam_beta1": 0.9,
  "adam_beta2": 0.95,
  "weight_decay": 0.1,
  "max_grad_norm": 1.0,
  "warmup_steps": 500,
  "lr_scheduler_type": "cosine",
  "max_steps": $WARMUP_MAX_STEPS,
  "save_steps": 1000,
  "eval_steps": 1000,
  "logging_steps": 10,
  "seed": 42,
  "recompute": true,
  "bf16": true,
  "fp16_opt_level": "O2",
  "dataloader_num_workers": 1,
  "distributed_dataloader": 1,
  "continue_training": 0,
  "do_train": true,
  "do_eval": false,
  "do_predict": false,
  "unified_checkpoint": true,
  "overwrite_output_dir": true,
  "stabilize_steps": 0,
  "postmix_warmup_steps": 0,
  "teacher_model_path": "$GQA_CONFIG",
  "teacher_checkpoint": "$GQA_CKPT",
  "distill_alpha": $WARMUP_DISTILL_ALPHA,
  "distill_temperature": 2.0,
  "layer_distill_beta": $WARMUP_LAYER_BETA,
  "distill_end_step": 0,
  "distill_decay_steps": 0
}
JSON
  echo "wrote $WARMUP_CONFIG"
}

run_warmup() {
  make_warmup_config
  cd "$VHA_DIR"
  mpirun bash scripts/train_warmup.sh "$WARMUP_CONFIG"
}

if [[ $# -ne 1 ]]; then usage; exit 1; fi
case "$1" in
  clean) rm -rf "$WORK_DIR" ;;
  collect) run_collect ;;
  convert) run_convert ;;
  eval-init) run_eval "$INIT_DIR" init ;;
  refine) run_refine ;;
  eval-refine) run_eval "$REFINE_DIR" refine ;;
  make-warmup-config) make_warmup_config ;;
  warmup) run_warmup ;;
  all) run_collect; run_convert; run_eval "$INIT_DIR" init; run_refine; run_eval "$REFINE_DIR" refine; make_warmup_config ;;
  *) usage; exit 1 ;;
esac
