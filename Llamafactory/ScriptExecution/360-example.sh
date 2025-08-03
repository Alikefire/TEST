export DS_SKIP_CUDA_CHECK=1 
export DISABLE_VERSION_CHECK=1  # if necessary
#使用 deepspeed 指令启动 DeepSpeed 引擎时您无法使用 CUDA_VISIBLE_DEVICES 指定GPU
export CUDA_VISIBLE_DEVICES=0,6,7       # 指定使用GPU
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1" 
export NCCL_DEBUG=INFO
export HF_TOKENIZER_PARALLELISM=false

LOG_DIR="logs"
TIMESTAMP=$(date "+%Y%m%d-%H%M%S")    # 时间戳格式

if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
  echo "📁 已创建日志目录: $LOG_DIR"
fi

# --------------------------
# 执行训练命令
# --------------------------
echo "🚀 开始训练..."
echo "🖥️  使用GPU: $CUDA_VISIBLE_DEVICES"
echo "📅 开始时间: $(date)"
# sft
# --include localhost:2,6,7
#--drop_exceed_length_data参数实际上没有作用，只有警告作用
# --flash_attn sdpa 可用后端只有sdpa，fa以及fa2
(deepspeed  src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path ./Model/OriginalModel/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --dataset s1.1 \
    --template deepseekr1 \
    --finetuning_type lora \
    --output_dir ./Model/saves/DeepSeek-R1-Distill-Qwen-1.5B-s1.1-mft/lora-0525 \
    --cache_dir .cache \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 25000 \
    --drop_exceed_length_data True \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 200 \
    --learning_rate 1e-5 \
    --num_train_epochs 9 \
    --plot_loss \
    --save_only_model True \
    --deepspeed examples/deepspeed/ds_z3_offload_config.json \
    --fp16 True \
    --flash_attn sdpa \
    --gradient_checkpointing True \
    --seed 42 \
    --sequence_parallel_size 3 \
    --packing True \
    --preprocessing_num_workers 16 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target q_proj,k_proj,v_proj,o_proj,gate_proj,down_proj,up_proj,lm_head,embed_tokens \
    --report_to tensorboard \
    --use_mft True \
    --mft_target_mask_probability 0.4\
    --mft_warmup_ratio 0.5
) 2>&1| tee "${LOG_DIR}/train_${TIMESTAMP}.log"
# dpo
# deepspeed --hostfile=hostfile.1mac src/train.py \
#     --stage dpo \
#     --do_train \
#     --model_name_or_path [your qwen3 model path] \
#     --dataset dpo_toy \
#     --template qwen3 \
#     --finetuning_type full \
#     --pref_beta 0.1 \
#     --pref_loss sigmoid \
#     --output_dir output/debug \
#     --cache_dir .cache \
#     --overwrite_cache \
#     --overwrite_output_dir \
#     --cutoff_len 32768 \
#     --per_device_train_batch_size 1 \
#     --gradient_accumulation_steps 4 \
#     --lr_scheduler_type cosine \
#     --warmup_ratio 0.0 \
#     --logging_steps 1 \
#     --save_steps 2000 \
#     --save_strategy steps \
#     --learning_rate 1e-6 \
#     --num_train_epochs 10 \
#     --plot_loss \
#     --save_only_model True \
#     --deepspeed examples/deepspeed/ds_z3_offload_config.json \
#     --flash_attn fa2 \
#     --gradient_checkpointing True \
#     --bf16 True \
#     --ddp_timeout 180000000 \
#     --seed 42 \
#     --sequence_parallel_size 4
