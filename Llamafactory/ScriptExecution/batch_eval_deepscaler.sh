#!/bin/bash
# 增强版批量任务脚本

# 要监控的程序名称或PID
TARGET_PROGRAM="src/train.py"  # 替换为您要监控的程序名
# 或者使用PID: TARGET_PID="12345"

# 检查程序是否还在运行
while pgrep -f "$TARGET_PROGRAM" > /dev/null; do
    echo "[$(date +'%T')] 程序 $TARGET_PROGRAM 仍在运行，等待中..."
    sleep 600  # 每分钟检查一次
done

echo "[$(date +'%T')] 程序 $TARGET_PROGRAM 已结束，开始执行评估脚本"

# 任务列表:"aime", "amc", "math", "minerva", "olympiad_bench","aime25","gpqa"
declare -a TASKS=("aime" "amc" "gpqa")
# 模型列表
declare -a MODELS=(
    # "./Model/OriginalModel/Qwen/Qwen3-0.6B"
    # "./Model/OriginalModel/Qwen/Qwen3-1.7B"
    # "./Model/MergeModel/Qwen3-1.7B-Mot_mix-sft-16.2k_dwi"
    # "./Model/OriginalModel/prithivMLmods/Capricornus-MoT-1.7B-Supreme1"
    # "./Model/OriginalModel/ertghiu256/qwen3-1.7b-mixture-of-thought"
    # "./Model/MergeModel/Qwen3-1.7B-Mot_mix-sft-16.2k_dwi-reasoning"
    # "./Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k_dwi-reasoning"
    # "./Model/MergeModel/Qwen3-1.7B-Mot_mix-sft-16.2k"
    # "./Model/OriginalModel/prithivMLmods/Theta-Crucis-0.6B-Turbo1"
    # "./Model/OriginalModel/prithivMLmods/Eta-Aurigae-0.6B-Echelon1"
    # "./Model/OriginalModel/prithivMLmods/Nenque-MoT-0.6B-Elite14"
    # "./Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k"
    # "./Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k_rw"
    # "./Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k_dwi"
    # "./Model/MergeModel/Qwen3-1.7B-Mot_mix-sft-16.2k-lora"
    # "./Model/MergeModel/Qwen3-0.6B-Mot_math-sft-84k/checkpoint-1200"
    "./Model/MergeModel/Qwen3-0.6B-Mot_mixed-sft-194.4k"
)
OUTPUT_DIR="./eval_result/deepscaler_results"
# 颜色定义（兼容不同终端）
COLOR_GREEN='\033[1;32m'
COLOR_RED='\033[1;31m'
COLOR_RESET='\033[0m'



echo "============ 开始执行批量任务  ============"

for model in "${MODELS[@]}"; do
    echo -e "\n${COLOR_GREEN}[$(date +'%T')] 开始评估模型: $model ${COLOR_RESET}"
    for task in "${TASKS[@]}"; do
        # 任务开始提示
        echo -e "\n${COLOR_GREEN}[$(date +'%T')] 开始执行任务: $task ${COLOR_RESET}"

        # 记录开始时间
        start_time=$(date +%s)
        # 获取当前日期和时间（精确到分钟）
        RUN_TIMESTAMP=$(date +'%Y%m%d-%H%M')

        # 构建包含日期和任务名称的最终输出目录路径
        task_output_dir="${OUTPUT_DIR}/${task}/${RUN_TIMESTAMP}"

        # 确保任务特定的输出目录存在
        mkdir -p "$task_output_dir"

        # 执行任务并捕获退出状态，添加 --output-dir 参数
        if ./Light-R1/deepscaler-release/scripts/eval/eval_model.sh --datasets "$task" --model "$model" --output-dir "$task_output_dir"; then
            # 计算耗时
            end_time=$(date +%s)
            duration=$((end_time - start_time))

            # 成功提示
            echo -e "${COLOR_GREEN}✓ [$(date +'%T')] 任务 $task 已完成 (耗时: ${duration}s) ${COLOR_RESET}"
        else
            # 错误处理
            echo -e "${COLOR_RED}✗ [$(date +'%T')] 任务 $task 执行失败 ${COLOR_RESET}"
            # 可选择是否中断后续任务
            # exit 1
        fi
    done
done
echo -e "\n${COLOR_GREEN}============ 所有任务执行完毕 ============${COLOR_RESET}"