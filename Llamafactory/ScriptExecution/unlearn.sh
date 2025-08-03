#!/bin/bash
# 增强型模型训练脚本
# 功能: 参数校验、日志记录、错误处理

# 配置区
LOG_FILE="./logs/unlearn_$(date +%Y%m%d_%H%M%S).log"
MODEL="./Model/OriginalModel/meta-llama/Llama-3.2-3B-Instruct"  # 默认模型
STRATEGY="sentencize" # 等号两边不能有空格
DATASET="sports"
LEARNINGRATE="3e-05"
METHOD="npo_KL"
MAINPROGRAME="./parametric-faithfulness/unlearn.py"

# 解析命令行参数，以便可以批量运行脚本
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --lr)
            LEARNINGRATE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --pos|--ff2|--stepwise)
            # 处理不需要值的标志参数
            shift
            ;;
        *)
            echo "[WARNING] 未知参数: $1"
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
export CUDA_VISIBLE_DEVICES=1,2,3      # 指定使用GPU 0
# 在运行脚本前设置环境变量
#export CUDA_LAUNCH_BLOCKING=1  # 强制同步执行,以便展示出完整的错误信息
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 初始化日志系统
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== 开始执行训练任务 [$(date)] ====="

# 参数校验
if [ ! -f "$MAINPROGRAME" ]; then
    echo "[ERROR] 主程序文件 $MAINPROGRAME 未找到" >&2
    exit 2
fi

# 核心执行模块
echo "[INFO] 加载模型: $MODEL"
echo "[DEBUG] 任务参数: --model_name=$MODEL --strategy=$STRATEGY --dataset=$DATASET --lr=$LEARNINGRATE --method=$METHOD --stepwise --pos --ff2"

python "$MAINPROGRAME" \
    --model_name "$MODEL" \
    --strategy "$STRATEGY" \
    --stepwise \
    --dataset "$DATASET" \
    --lr "$LEARNINGRATE" \
    --pos \
    --ff2 \
    --method "$METHOD"

# 状态码处理
case $? in
    0)
        echo "[SUCCESS] 训练任务正常完成"
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