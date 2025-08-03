#!/bin/bash
# 增强型模型评估脚本
# 功能: 参数校验、日志记录、错误处理

# 配置区
LOG_FILE="./logs/livecode_eval_$(date +%Y%m%d_%H%M%S).log"
MODEL="../Model/OriginalModel/Qwen/Qwen3-0.6B"  # 默认模型
MODEL_NAME="Qwen/Qwen3-0.6B" #"prithivMLmods/Theta-Crucis-0.6B-Turbo1" "prithivMLmods/Nenque-MoT-0.6B-Elite14" "prithivMLmods/Eta-Aurigae-0.6B-Echelon1"
TASK="release_v4"  # v1,v2,v3,v4,v5

# 解析命令行参数，以便可以批量运行脚本
while [[ $# -gt 0 ]]; do
    case $1 in
        --release_version)
            TASK="$2"
            shift 2
            ;;
        --local_model_path)
            MODEL="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# --------------------------
# 环境变量配置
# --------------------------
#设置环境变量使lighteval库从镜像库下载任务数据集
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_DATASETS_CACHE="/home/zdd/xx_help/LLaMA-Factory/data/huggingface/datasets"  # 可选：指定数据集缓存位置
export CUDA_VISIBLE_DEVICES=0,1,2,3     # 指定使用GPU 0
# 在运行脚本前设置环境变量
#export CUDA_LAUNCH_BLOCKING=1  # 强制同步执行,以便展示出完整的错误信息
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 初始化日志系统
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== 开始执行评估任务 [$(date)] ====="


# 核心执行模块
echo "[INFO] 加载模型: $MODEL"
echo "[DEBUG] 任务参数: --task==$TASK"
#accelerate launch进行分布式训练确保显卡的负载均衡
#--use_cache 缓存输出结果，结合--continue_existing来使用
#--continue_existing \ 使用之前已回答的问题的结果
python -m lcb_runner.runner.main \
    --local_model_path "$MODEL" \
    --model "$MODEL_NAME" \
    --release_version "$TASK" \
    --scenario codegeneration \
    --cot_code_execution \
    --temperature 0.6 \
    --top_p 0.90 \
    --max_tokens 38912 \
    --n 4 \
    --use_cache \
    --continue_existing \
    --cache_batch_size 200 \
    --evaluate 


# 状态码处理
case $? in
    0)
        echo "[SUCCESS] 评估任务正常完成"
        ;;
    124)
        echo "[TIMEOUT] 执行超时（600秒限制）" >&2
        exit 3
        ;;
    *)
        echo "[CRITICAL] 非预期错误，错误码: $?" >&2
        exit 4
        ;;
esac

echo "===== 任务结束 [$(date)] ====="