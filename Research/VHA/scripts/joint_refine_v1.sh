#!/bin/bash
# Joint refine v1: single-stage replacement of cascade -> joint_logit pipeline.
# Loss = alpha*block_MSE + beta*logit_KL + gamma*final_hidden_MSE with annealing.
# Plateau-based termination instead of fixed step count.
# Starts from the clean PCA-init (kv_postmix_activation_128) — this single stage
# fully replaces alignment + cascade + joint_logit (DHA-style fuse-from-init).
set -e

source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate
export PYTHONPATH=$(pwd)/../

unset PADDLE_ELASTIC_JOB_ID PADDLE_TRAINER_ENDPOINTS DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT PADDLE_ELASTIC_TIMEOUT PADDLE_TRAINER_ID
unset OUTPUT_PATH SYS_OUTPUT_PATH COMBINED_OUTPUT_PATH

bash scripts/kill_process.sh 2>/dev/null || true

EXP_NAME=${EXP_NAME:-qwen3_vha_1p7B_joint_refine_v1}
LOG_DIR=output/${EXP_NAME}_log
mkdir -p $LOG_DIR

GQA_CHECKPOINT=${GQA_CHECKPOINT:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/output/qwen3_gqa_1p7B_pretrain/checkpoint-24000/model_state_merged}
GQA_MODEL_CONFIG=${GQA_MODEL_CONFIG:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA/config/qwen3/Qwen3-1.7B-GQA}
# Start from the clean PCA-init checkpoint (kv_postmix_activation_128).
# Joint_refine fully replaces alignment + cascade + joint_logit so we test it
# from the truly-untrained init, matching DHA's "fuse from raw init via single
# L_fusion" recipe. Arch (16Q + 2KV + postmix r=4) is identical to attn8.
VHA_CHECKPOINT=${VHA_CHECKPOINT:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_kv_postmix_activation_128}
VHA_MODEL_CONFIG=${VHA_MODEL_CONFIG:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/qwen3_vha_1p7B_kv_postmix_activation_128}
OUTPUT_PATH=${OUTPUT_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/PaddleFleet/Research/VHA-Warmup/output/${EXP_NAME}}
DATA_PATH=${DATA_PATH:-/root/paddlejob/share-storage/gpfs/system-public/dingxibo/datasets/fineweb-edu/qwen}

GPUS=${GPUS:-0,1,2,3,4,5,6,7}
# Generous data; plateau detection will stop us when convergence flattens.
# 65536 samples * 2048 tokens = 134M tokens (8x v3 joint_logit data).
# Each rank loads only its 8192-sample shard (~135MB host RAM/rank).
NUM_SAMPLES=${NUM_SAMPLES:-65536}
SEQ_LENGTH=${SEQ_LENGTH:-2048}

# Hard cap; plateau usually stops earlier
MAX_STEPS=${MAX_STEPS:-15000}
MIN_STEPS=${MIN_STEPS:-1000}

LR=${LR:-3e-5}
WARMUP_STEPS=${WARMUP_STEPS:-200}
LR_MIN_RATIO=${LR_MIN_RATIO:-0.05}
GRAD_CLIP=${GRAD_CLIP:-1.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}
ADAM_BETA2=${ADAM_BETA2:-0.95}
# bs=4 fits comfortably with 1.7B + bf16 + attn-only optim states.
# Effective global batch = 4 * 8 GPUs = 32.
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
KL_TEMPERATURE=${KL_TEMPERATURE:-1.0}
TRAIN_MODE=${TRAIN_MODE:-attn}
LOG_INTERVAL=${LOG_INTERVAL:-20}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}

# Loss schedule (linear interp over ANNEAL_STEPS steps)
ALPHA_START=${ALPHA_START:-1.0}
ALPHA_END=${ALPHA_END:-0.1}
BETA_START=${BETA_START:-0.1}
BETA_END=${BETA_END:-1.0}
GAMMA_START=${GAMMA_START:-0.0}
GAMMA_END=${GAMMA_END:-0.2}
ANNEAL_STEPS=${ANNEAL_STEPS:-2000}

# Plateau detection: stop if rolling 300-step best improves <0.5% vs prior 300
PLATEAU_WINDOW=${PLATEAU_WINDOW:-300}
PLATEAU_EPS=${PLATEAU_EPS:-0.005}
PLATEAU_METRIC=${PLATEAU_METRIC:-kl}

python -m paddle.distributed.launch \
    --gpus="$GPUS" \
    --log_dir "$LOG_DIR" \
    ../VHA-Warmup/scripts/joint_refine.py \
    --gqa_checkpoint "$GQA_CHECKPOINT" \
    --gqa_model_config "$GQA_MODEL_CONFIG" \
    --vha_checkpoint "$VHA_CHECKPOINT" \
    --vha_model_config "$VHA_MODEL_CONFIG" \
    --output_path "$OUTPUT_PATH" \
    --data_path "$DATA_PATH" \
    --num_samples "$NUM_SAMPLES" \
    --seq_length "$SEQ_LENGTH" \
    --max_steps "$MAX_STEPS" \
    --min_steps "$MIN_STEPS" \
    --lr "$LR" \
    --warmup_steps "$WARMUP_STEPS" \
    --lr_min_ratio "$LR_MIN_RATIO" \
    --grad_clip "$GRAD_CLIP" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta2 "$ADAM_BETA2" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --kl_temperature "$KL_TEMPERATURE" \
    --train_mode "$TRAIN_MODE" \
    --log_interval "$LOG_INTERVAL" \
    --save_interval "$SAVE_INTERVAL" \
    --alpha_start "$ALPHA_START" \
    --alpha_end "$ALPHA_END" \
    --beta_start "$BETA_START" \
    --beta_end "$BETA_END" \
    --gamma_start "$GAMMA_START" \
    --gamma_end "$GAMMA_END" \
    --anneal_steps "$ANNEAL_STEPS" \
    --plateau_window "$PLATEAU_WINDOW" \
    --plateau_eps "$PLATEAU_EPS" \
    --plateau_metric "$PLATEAU_METRIC"
