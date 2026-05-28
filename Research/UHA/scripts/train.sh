#!/bin/bash
source /root/paddlejob/share-storage/gpfs/system-public/dingxibo/venv_paddlefleet/bin/activate

export PYTHONPATH=$(pwd):$(pwd)/../

#mpi_rank=${OMPI_COMM_WORLD_RANK:-0}
#node_rank=$((mpi_rank+offset))
#mpi_node=${OMPI_COMM_WORLD_SIZE:-1}
#echo "MPI status:${mpi_rank}/${mpi_node}"
#nnode_train=${nnode_set:-${mpi_node}}
#master_train=${master:-localhost}
##
#echo "Distributed Training ${node_rank}/${nnode_train} master=${master_train}"
#set -x
#
# 屏蔽平台预设的环境变量，因为框架采用兼容升级，检测到这些配置会使用原方式启动
unset PADDLE_ELASTIC_JOB_ID
unset PADDLE_TRAINER_ENDPOINTS
unset DISTRIBUTED_TRAINER_ENDPOINTS
unset FLAGS_START_PORT
unset PADDLE_ELASTIC_TIMEOUT
unset PADDLE_TRAINER_ID
#
## 保证集群稳定性的配置，跟性能无关
#export NCCL_IB_QPS_PER_CONNECTION=8 
#export NCCL_IB_TIMEOUT=22
#export NCCL_IB_GID_INDEX=3
#export NCCL_NVLS_ENABLE=0
## 开启AR功能
#export NCCL_IB_ADAPTIVE_ROUTING=1
#
## 使用BCCL，需要配合镜像版本 >= FleetY10.1.0
#export LD_LIBRARY_PATH=/usr/local/bccl/lib:$LD_LIBRARY_PATH
#
## 增加tcp_syn_max_backlog, 避免建联失败
#export FLAGS_tcp_max_syn_backlog=16384
#
## 保证先launch的kernel先抢占SM, 提高tpsp_comm_overlap效率
#export CUDA_DEVICE_MAX_CONNECTIONS=1
#
## 错误发生时打印更全的堆栈信息
#export FLAGS_call_stack_level=2

SCRIPT_DIR=`dirname "$0"`
LAUNCH_CMD=`python $SCRIPT_DIR/selective_launch.py 36677 `
if [[ -z "$LAUNCH_CMD" ]]; then
    exit 0
fi

sh scripts/kill_process.sh

RANK=`echo ${LAUNCH_CMD} | grep -oP '(?<=--rank )\d+'`

EXP_NAME=$1
shift 1
LOG_DIR=output/$EXP_NAME/trainer-${RANK}
mkdir -p $LOG_DIR

python -m paddle.distributed.launch \
    --log_dir $LOG_DIR \
    $LAUNCH_CMD \
    --run_mode=collective \
    ${script:-run_pretrain.py}  \
    $@
