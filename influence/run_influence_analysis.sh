#!/bin/bash

# 训练执行脚本: train_ppo.sh
# 用法: ./train_ppo.sh

# --------------------------
# 配置区（按需修改）
# --------------------------
# 设置模型、数据集和保存路径
MODEL_PATH="./Model/OriginalModel/Qwen/Qwen2.5-0.5B-Instruct" # 替换为您的模型路径
FULL_TRAIN_DATASET="./output_per_dataset_analysis/processed_splits/train" # 替换为您的完整训练数据集路径
VALIDATION_DATASET="./output_per_dataset_analysis/processed_splits/validation" # 替换为您的验证数据集路径
SAVE_PATH="./TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short" # 替换为您的结果保存路径
LOG_DIR="logs"                        # 日志存放目录
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")    # 时间戳格式
LOG_FILE="${LOG_DIR}/influence_train_${TIMESTAMP}.log"

# 设置分布式训练的GPU数量
# NUM_GPUS=1 # 根据您的GPU数量进行修改，例如：2, 4, 8
# --------------------------
# 环境变量配置
# --------------------------
export CUDA_VISIBLE_DEVICES=1,3,7      # 指定使用GPU 
export NCCL_P2P_DISABLE=1             # 禁用NCCL P2P通信
export NCCL_IB_DISABLE=1              # 禁用NCCL InfiniBand
export FORCE_TORCHRUN=1               #强制使用 torchrun 而不是 torch.distributed.launch
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #缓解内存碎片化

# --------------------------
# 预检查
# --------------------------
# 检查保存路径是否存在，如果不存在则创建
if [ ! -d "$SAVE_PATH" ]; then
  mkdir -p "$SAVE_PATH"
  echo "Created directory: $SAVE_PATH"
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
#sub-train 保持和val子集目录顺序一致
accelerate launch \
  ./TEST/influence/main.py \
  --model-path "$MODEL_PATH" \
  --full-train "$FULL_TRAIN_DATASET" \
  --validation-path "$VALIDATION_DATASET" \
  --save-path "$SAVE_PATH" \
  --use-full-layer False \
  --target-layers "model.layers.1.mlp.gate_proj" "model.layers.5.mlp.gate_proj" "model.layers.10.mlp.gate_proj" "model.layers.15.mlp.gate_proj" "model.layers.20.mlp.gate_proj" "model.layers.24.mlp.gate_proj" "model.layers.25.mlp.gate_proj" "model.layers.26.mlp.gate_proj" "model.layers.27.mlp.gate_proj" "model.layers.28.mlp.gate_proj" \
  --without-output False \
  --without-attention False \
  --sub-train "gsm8k-gpt4o_train" "gsm8k-r1_train" "s1K-mix_s1_brief_cot"

# --------------------------
# 结果检查
# --------------------------
if [ $? -eq 0 ]; then
  echo "✅ 训练成功完成!"
  echo "📅 结束时间: $(date)"
else
  echo " 训练失败！请检查日志: $LOG_FILE"
  exit 1
fi