#!/bin/bash
# 增强型模型评估脚本
# 功能: 参数校验、日志记录、错误处理

# 配置区
LOG_FILE="./logs/math_eval_$(date +%Y%m%d_%H%M%S).log"
MODEL="./Model/OriginalModel/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # 默认模型
TASK="aime24"  # 默认任务
INPUTFILE="examples/sample_answers.csv"
OUTPUTFILE="output.csv"
#评估用evaluate_model.py，格式化csv文件用evaluate_model_outputs.py，提取答案用extract_answers.py
MAINPROGRAME="./Math-Verify/evaluate_model.py"
BATCHSIZE=1

# 解析命令行参数，以便可以批量运行脚本
while [[ $# -gt 0 ]]; do
    case $1 in
        --task)
            TASK="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
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
export CUDA_VISIBLE_DEVICES=0,1,2,3      # 指定使用GPU 0
# 在运行脚本前设置环境变量
#export CUDA_LAUNCH_BLOCKING=1  # 强制同步执行,以便展示出完整的错误信息
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 初始化日志系统
exec > >(tee -a "$LOG_FILE") 2>&1

echo "===== 开始执行评估任务 [$(date)] ====="

# 参数校验
if [ ! -f "$MAINPROGRAME" ]; then
    echo "[ERROR] 主程序文件 $MAINPROGRAME 未找到" >&2
    exit 2
fi

# 核心执行模块
echo "[INFO] 加载模型: $MODEL"
echo "[DEBUG] 任务参数: --task==$TASK"
#accelerate launch进行分布式训练确保显卡的负载均衡
accelerate launch "$MAINPROGRAME" \
    --model "$MODEL" \
    --task "$TASK" 
    # --use_chat_template \
    # --override_bs "$BATCHSIZE"
    #python evaluate_model_outputs.py \
    # --input_csv "INPUTFILE" \
    # --output_csv "OUTPUTFILE"

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