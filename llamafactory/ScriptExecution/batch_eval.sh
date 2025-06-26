#!/bin/bash
# 增强版批量任务脚本

# 任务列表:"aime24" "math_hard" "gsm8k" "amc23" "math_500"
declare -a TASKS=( "aime24" "amc23")
# 模型列表
declare -a MODELS=(
    "/home/zdd/xx_help/MaskedThought/my_deepseek_1.5b_distill_gsm8k/checkpoint-3495"
    "./Model/OriginalModel/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
)

# 颜色定义（兼容不同终端）
COLOR_GREEN='\033[1;32m'
COLOR_RED='\033[1;31m'
COLOR_RESET='\033[0m'

echo "============ 开始执行批量任务 ============"
for model in "${MODELS[@]}"; do
    echo -e "\n${COLOR_GREEN}[$(date +'%T')] 开始评估模型: $model ${COLOR_RESET}"
    for task in "${TASKS[@]}"; do
        # 任务开始提示
        echo -e "\n${COLOR_GREEN}[$(date +'%T')] 开始执行任务: $task ${COLOR_RESET}"
        
        # 记录开始时间
        start_time=$(date +%s)
        
        # 执行任务并捕获退出状态
        if ./ScriptExecution/math_eval.sh --task "$task" --model "$model"; then
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