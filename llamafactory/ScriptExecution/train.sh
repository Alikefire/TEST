#!/bin/bash

# 训练执行脚本: train_ppo.sh
# 用法: ./train_ppo.sh

# --------------------------
# 配置区（按需修改）
# --------------------------
YAML_CONFIG="YamlSet/SFT_s1.1.yaml"  # 配置文件路径
LOG_DIR="logs"                        # 日志存放目录
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")    # 时间戳格式

# --------------------------
# 环境变量配置
# --------------------------
#使用 deepspeed 指令启动 DeepSpeed 引擎时您无法使用 CUDA_VISIBLE_DEVICES 指定GPU
export CUDA_VISIBLE_DEVICES=0,2,7      # 指定使用GPU 
export NCCL_P2P_DISABLE=1             # 禁用NCCL P2P通信
export NCCL_IB_DISABLE=1              # 禁用NCCL InfiniBand
# export FORCE_TORCHRUN=1               #强制使用 torchrun 而不是 torch.distributed.launch
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

# --------------------------
# 执行训练命令
# --------------------------
echo "🚀 开始训练..."
echo "🖥️  使用GPU: $CUDA_VISIBLE_DEVICES"
echo "📄 配置文件: $YAML_CONFIG"
echo "📅 开始时间: $(date)"

# 执行命令并记录日志（tee命令同时输出到终端和文件）
# llamafactory-cli train "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"
#合并模型
# llamafactory-cli export "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"
#推理
# llamafactory-cli chat "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/chat_${TIMESTAMP}.log"
#评估
# llamafactory-cli eval "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"
#使用deepspeed 训练,使用-include localhost:0,2,7指定GPU
# 指定最大sample数，--max_samples 1000 \
# 开启流式加载 --streaming True \
# 和流式加载一起使用--max_steps 400 \ ， --packing True \packing策略有时会有有问题
#  --preprocessing_num_workers 1 \加快预处理耗时
(deepspeed  --include localhost:0,1,3 src/train.py --model_name_or_path ./Model/OriginalModel/Qwen/Qwen2.5-0.5B-Instruct \
  --trust_remote_code True \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --dataset mix_of_thought_instruct_code_4k,mix_of_thought_instruct_math_4k,mix_of_thought_instruct_science_4k \
  --template qwen \
  --cutoff_len 32767 \
  --overwrite_cache True \
  --dataloader_num_workers 4 \
  --output_dir ./Model/MergeModel/Qwen2.5-0.5B-Instruc-Mot_mix-sft-4k \
  --logging_steps 10 \
  --save_steps 5 \
  --plot_loss true \
  --overwrite_output_dir True \
  --save_only_model False \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --learning_rate 1.0e-5 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --fp16 True \
  --ddp_timeout 180000000 \
  --max_grad_norm 1 \
  --flash_attn sdpa \
  --gradient_checkpointing True \
  --use_unsloth_gc True \
  --torch_empty_cache_steps 1000 \
  --seed 42 \
  --drop_exceed_length_data True \
  --deepspeed examples/deepspeed/ds_z3_offload_config.json \
  --sequence_parallel_size 3 \
  --max_steps 86 \
  --packing True \
  --streaming True \
  --report_to tensorboard) 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"

# --------------------------
# 结果检查
# --------------------------
if [ $? -eq 0 ]; then
  echo "✅ 训练成功完成!"
  echo "📅 结束时间: $(date)"
else
  echo "❌ 训练失败！请检查日志: ${LOG_DIR}/ppo_train_${TIMESTAMP}.log"
  exit 1
fi