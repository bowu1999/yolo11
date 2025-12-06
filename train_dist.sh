#!/bin/bash

# 设置要使用的 GPU
all_gpu_ids=$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ',' | sed 's/,$//')
script=$1
config=$2
if [ -z "$3" ]; then
    GPUS=$all_gpu_ids
    echo "[LOG]: No visible GPUs specified, using all available GPUs: $GPUS"
else
    GPUS=$3
    echo "[LOG]: Visible GPUs: $GPUS"
fi

IFS=',' read -r -a array <<< "$GPUS"

nproc_per_node=${#array[@]}

export PYTHONWARNINGS="ignore::DeprecationWarning"

# 使用 torchrun 启动分布式训练
CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$nproc_per_node -m $script --config $config