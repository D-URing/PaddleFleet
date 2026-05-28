#!/bin/bash
# VHA Warmup Training Launch Script
#
# Usage:
#   mpirun bash scripts/train_warmup.sh <config.json>
#
# Example:
#   mpirun bash scripts/train_warmup.sh config/qwen3/qwen3_vha_1p7B_warmup.json

source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate

export PYTHONPATH=$(pwd)/../

unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ID

SCRIPT_DIR=`dirname "$0"`
LAUNCH_CMD=`python $SCRIPT_DIR/selective_launch.py 36677 `
if [[ -z "$LAUNCH_CMD" ]]; then
    exit 0
fi

sh scripts/kill_process.sh

RANK=`echo ${LAUNCH_CMD} | grep -oP '(?<=--rank )\d+'`

CONFIG=$1
EXP_NAME=$(basename "$CONFIG" .json)
LOG_DIR=output/$EXP_NAME/trainer-${RANK}
mkdir -p $LOG_DIR

python -m paddle.distributed.launch \
    --log_dir $LOG_DIR \
    $LAUNCH_CMD \
    --run_mode=collective \
    run_warmup.py \
    "$CONFIG"
