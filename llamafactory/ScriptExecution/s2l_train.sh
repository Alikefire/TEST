#!/bin/bash

# 训练执行脚本: train_ppo.sh
# 用法: ./train_ppo.sh

# --------------------------
# 配置区（按需修改）
# --------------------------
YAML_CONFIG="configs/pythia-70m-deduped_checkpoints.yml"  # 配置文件路径
LOG_DIR="logs"                        # 日志存放目录
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")    # 时间戳格式
LOG_FILE="${LOG_DIR}/s2l_train_${TIMESTAMP}.log"
WANDB_KEY="be3827dada95edbcf0fb39f0578c548340baf3f5"
# --------------------------
# 环境变量配置
# --------------------------
export CUDA_VISIBLE_DEVICES=5,6,7       # 指定使用GPU 
export NCCL_P2P_DISABLE=1             # 禁用NCCL P2P通信
export NCCL_IB_DISABLE=1              # 禁用NCCL InfiniBand
export FORCE_TORCHRUN=1               #强制使用 torchrun 而不是 torch.distributed.launch
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# --------------------------
# 预检查
# --------------------------
if [ ! -f "$YAML_CONFIG" ]; then
  echo "❌ 错误：配置文件 $YAML_CONFIG 未找到"
  exit 1
fi

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
  echo "📁 已创建日志目录: $LOG_DIR"
fi

# 初始化日志系统
exec > >(tee -a "$LOG_FILE") 2>&1
# --------------------------
# 执行训练命令
# --------------------------
echo "🚀 开始训练..."
echo "🖥️  使用GPU: $CUDA_VISIBLE_DEVICES"
echo "📄 配置文件: $YAML_CONFIG"
echo "📅 开始时间: $(date)"

# --nproc_per_node=3 ，设置可用GPU数目
# 执行命令并记录日志（tee命令同时输出到终端和文件）
nohup torchrun  train.py \
    --config_file $YAML_CONFIG \
    --wandb_key $WANDB_KEY 

# --------------------------
# 结果检查
# --------------------------
if [ $? -eq 0 ]; then
  echo "✅ 训练成功完成!"
  echo "📅 结束时间: $(date)"
else
  echo "❌ 训练失败！请检查日志: $LOG_FILE"
  exit 1
fi