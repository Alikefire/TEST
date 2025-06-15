#!/bin/bash

# 训练执行脚本: train_ppo.sh
# 用法: ./train_ppo.sh

# --------------------------
# 配置区（按需修改）
# --------------------------
YAML_CONFIG="../S2L/configs/qwen2.5-0.5b_long-short_checkpoint.yml"  # 配置文件路径
LOG_DIR="logs"                        # 日志存放目录
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")    # 时间戳格式
LOG_FILE="${LOG_DIR}/s2l_train_${TIMESTAMP}.log"
# WANDB_KEY="be3827dada95edbcf0fb39f0578c548340baf3f5"
MODEL_PATH="./Model/MergeModel/DeepSeek-R1-Distill-Qwen-0.5B-long_short-sft-more_ckpt"
#需要修改model_path为check所在目录
# --------------------------
# 环境变量配置
# --------------------------
export CUDA_VISIBLE_DEVICES=0,1,2,6      # 指定使用GPU 
export NCCL_P2P_DISABLE=1             # 禁用NCCL P2P通信
export NCCL_IB_DISABLE=1              # 禁用NCCL InfiniBand
export FORCE_TORCHRUN=1               #强制使用 torchrun 而不是 torch.distributed.launch
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #缓解内存碎片化

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

# 执行命令并记录日志（tee命令同时输出到终端和文件）
#分布式训练使用torchrun 或 accelerate launch启动。单卡使用python启动。改到了在内层使用accelerate launch
python TEST/loss_path/run_distributed_trajectories.py \
    --model_path ${MODEL_PATH} \
    --config_file  ${YAML_CONFIG} \
    --checkpoints all

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