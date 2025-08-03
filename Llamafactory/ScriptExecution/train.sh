#!/bin/bash

# 训练执行脚本: train_ppo.sh
# 用法: ./train_ppo.sh

# --------------------------
# 配置区（按需修改）
# --------------------------
YAML_CONFIG="./YamlSet/merge_config.yaml"  # 配置文件路径
LOG_DIR="logs"                        # 日志存放目录
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")    # 时间戳格式

# --------------------------
# 环境变量配置
# --------------------------
#使用 deepspeed 指令启动 DeepSpeed 引擎时您无法使用 CUDA_VISIBLE_DEVICES 指定GPU
export TZ='Asia/Shanghai'            #时区是东八区
export CUDA_VISIBLE_DEVICES=0,1,2,3      # 指定使用GPU 
export NCCL_P2P_DISABLE=1             # 禁用NCCL P2P通信
export NCCL_IB_DISABLE=1              # 禁用NCCL InfiniBand
# export FORCE_TORCHRUN=1               #强制使用 torchrun 而不是 torch.distributed.launch
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #缓解内存碎片化
export MASTER_PORT=29501                                #在分布式训练时的29500端口被占用了时使用
export DISABLE_VERSION_CHECK=1                          #跳过llamafactory版本检查
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
#使用当前目录下的llamafactory源码,在src目录下执行
# python -m llamafactory.cli export  "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"
#推理
# llamafactory-cli chat "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/chat_${TIMESTAMP}.log"
#评估
# llamafactory-cli eval "$YAML_CONFIG" 2>&1 | tee "${LOG_DIR}/eval_${TIMESTAMP}.log"
#使用deepspeed 训练,使用-include localhost:0,2,7指定GPU
# 指定最大sample数，--max_samples 1000 \
# 开启流式加载 --streaming True \
# 和流式加载一起使用--max_steps 400 \ ， --packing True \packing策略有时会有有问题
#  --preprocessing_num_workers 1 \加快预处理耗时
#  --eval_dataset \
#  --resume_from_checkpoint ./Model/MergeModel/Qwen2.5-0.5B-Instruc-Mot_mix-sft-6k/checkpoint-80 \ 从断点接续训练
#多个dataset之间只用逗号隔开，不需要再用空格隔开
#--torch_empty_cache_steps 100 \每100步清理一次显存缓存，根据总步数来确定
#  --eval_delay 0.2 \
# --eval_strategy steps \
#  --eval_steps 150 \在训练过程中eval对显存压力较大,酌情考虑使用
#  --do_train \
# (deepspeed  --master_port 29501 --include localhost:0,1,2,3 src/train.py --model_name_or_path ./Model/OriginalModel/Qwen/Qwen3-1.7B \
#   --trust_remote_code True \
#   --stage sft \
#   --do_train \
#   --finetuning_type lora \
#   --lora_rank 32 \
#   --lora_alpha 32 \
#   --lora_dropout 0.05 \
#   --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj \
#   --learning_rate 1.0e-4 \
#   --dataset mix_of_thought_instruct_code_5.4k,mix_of_thought_instruct_math_5.4k,mix_of_thought_instruct_science_5.4k \
#   --template qwen3 \
#   --cutoff_len 32767 \
#   --overwrite_cache True \
#   --dataloader_num_workers 4 \
#   --output_dir ./Model/saves/Qwen3-1.7B-Mot_mix-sft-16.2k \
#   --logging_steps 200 \
#   --save_steps 600 \
#   --plot_loss true \
#   --overwrite_output_dir True \
#   --save_only_model False \
#   --per_device_train_batch_size 2 \
#   --gradient_accumulation_steps 3 \
#   --num_train_epochs 3.0 \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.05 \
#   --bf16 True \
#   --ddp_timeout 180000000 \
#   --max_grad_norm 1 \
#   --flash_attn sdpa \
#   --gradient_checkpointing True \
#   --use_unsloth_gc True \
#   --torch_empty_cache_steps 100 \
#   --seed 42 \
#   --drop_exceed_length_data True \
#   --deepspeed examples/deepspeed/ds_z3_offload_config.json \
#   --sequence_parallel_size 2 \
#   --max_steps 675 \
#   --packing True \
#   --streaming True \
#   --report_to tensorboard) 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"

# --------------------------
# 结果检查
# --------------------------
if [ $? -eq 0 ]; then
  echo "✅ 训练成功完成!"
  echo "📅 结束时间: $(date)"
#  --do_train \
(deepspeed  --master_port 29501 --include localhost:0,1,2,3 src/train.py --model_name_or_path ./Model/OriginalModel/Qwen/Qwen3-0.6B \
  --trust_remote_code True \
  --stage sft \
  --do_train \
  --finetuning_type full \
  --learning_rate 1.0e-5 \
  --dataset mix_of_thought_instruct_mixed \
  --resume_from_checkpoint ./Model/MergeModel/Qwen3-0.6B-Mot_mixed-sft-194.4k/checkpoint-900 \
  --template qwen3 \
  --cutoff_len 32767 \
  --overwrite_cache True \
  --dataloader_num_workers 4 \
  --output_dir ./Model/MergeModel/Qwen3-0.6B-Mot_mixed-sft-194.4k \
  --logging_steps 100 \
  --save_steps 900 \
  --plot_loss true \
  --overwrite_output_dir True \
  --save_only_model False \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 3 \
  --num_train_epochs 3.0 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  --bf16 True \
  --ddp_timeout 180000000 \
  --max_grad_norm 1 \
  --flash_attn sdpa \
  --gradient_checkpointing True \
  --use_unsloth_gc True \
  --torch_empty_cache_steps 100 \
  --seed 42 \
  --drop_exceed_length_data True \
  --deepspeed examples/deepspeed/ds_z3_offload_config.json \
  --sequence_parallel_size 2 \
  --max_steps 8100 \
  --packing True \
  --streaming True \
  --report_to tensorboard) 2>&1 | tee "${LOG_DIR}/train_${TIMESTAMP}.log"
else
  echo "❌ 训练失败！请检查日志: ${LOG_DIR}/ppo_train_${TIMESTAMP}.log"
  exit 1
fi