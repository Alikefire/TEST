#!/bin/bash

# 简化版本 - 批量scp传输指定checkpoint的losses.pt文件

REMOTE_HOST="10.72.74.13"
REMOTE_USER="zdd"
PORT=10022
MODEL_PATH="DeepSeek-R1-Distill-Qwen-0.5B-s1.1_mix-sft-more_ckpt"
CHECKPOINTS="6 12 18 24 30 36 42 48 54 60 66 72 78 84 87"

# 修正：远程服务器的实际路径
REMOTE_BASE_PATH="/home/zdd/xx_help/LLaMA-Factory/Model/MergeModel"
# 本地路径保持不变
LOCAL_BASE_PATH="/home/xiexin/xx_help/LLaMA-Factory/Model/MergeModel"

REMOTE_FULL_PATH="$REMOTE_BASE_PATH/$MODEL_PATH"
LOCAL_FULL_PATH="$LOCAL_BASE_PATH/$MODEL_PATH"


echo "开始传输losses.pt文件..."
echo "远程路径: $REMOTE_FULL_PATH"
echo "本地路径: $LOCAL_FULL_PATH"
echo "Checkpoint编号: $CHECKPOINTS"
echo "========================================"

for checkpoint_num in $CHECKPOINTS; do
    remote_file="$REMOTE_FULL_PATH/checkpoint-$checkpoint_num/losses.pt"
    local_file="$LOCAL_FULL_PATH/checkpoint-$checkpoint_num/losses.pt"
    local_dir="$(dirname "$local_file")"
    
    echo "正在传输 checkpoint-$checkpoint_num..."
    
    # 创建本地目录
    mkdir -p "$local_dir"
    
    # 使用sshpass执行scp（需要先安装sshpass）
    if scp -P "$PORT" "$REMOTE_USER@$REMOTE_HOST:$remote_file" "$local_file"; then
        echo "✓ 成功: checkpoint-$checkpoint_num"
    else
        echo "✗ 失败: checkpoint-$checkpoint_num"
    fi
    echo ""
done

echo "传输完成!"