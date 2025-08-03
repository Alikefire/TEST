#!/bin/bash
# 增强版批量任务脚本

# 要监控的程序名称或PID
TARGET_PROGRAM="eval_model.sh "  # 替换为您要监控的程序名
# 或者使用PID: TARGET_PID="12345"

# 检查程序是否还在运行
while pgrep -f "$TARGET_PROGRAM" > /dev/null; do
    echo "[$(date +'%T')] 程序 $TARGET_PROGRAM 仍在运行，等待中..."
    sleep 60  # 每分钟检查一次
done

echo "[$(date +'%T')] 程序 $TARGET_PROGRAM 已结束，开始执行评估脚本"

# 任务列表:# v1,v2,v3,v4,v5
declare -a TASKS=("release_v4")
# 模型列表
declare -a MODELS=(
    # "Qwen/Qwen3-0.6B"
    # "prithivMLmods/Theta-Crucis-0.6B-Turbo1"
    # "prithivMLmods/Nenque-MoT-0.6B-Elite14"
    # "prithivMLmods/Eta-Aurigae-0.6B-Echelon1"
    # "Qwen3-0.6B-Mot_mix-sft-16.2k"
    # "Qwen3-0.6B-Mot_mix-sft-16.2k_dwi"
    # "Qwen3-0.6B-Mot_mix-sft-16.2k_rw"
    "ertghiu256/qwen3-1.7b-mixture-of-thought"
    
)
OUTPUT_DIR="./eval_result/livecode_results"
LOCAL_MODEL_FOLDER="../Model/OriginalModel"
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
        # # 获取当前日期和时间（精确到分钟）
        # RUN_TIMESTAMP=$(date +'%Y%m%d-%H%M')

        # # 构建包含日期和任务名称的最终输出目录路径
        # task_output_dir="${OUTPUT_DIR}/${task}/${RUN_TIMESTAMP}"

        # # 确保任务特定的输出目录存在
        # mkdir -p "$task_output_dir"

        # 执行任务并捕获退出状态，添加 --output-dir 参数
        if ../ScriptExecution/livecodeEval.sh --release_version "$task" --local_model_path "${LOCAL_MODEL_FOLDER}/$model" --model "$model"; then
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