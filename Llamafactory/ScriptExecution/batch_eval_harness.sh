#!/bin/bash
# 增强版批量任务脚本

export HF_ALLOW_CODE_EVAL="1"
# export HF_HUB_OFFLINE=1 #使用gpu时离线运行以避免获取huggingface资源时卡死
# 任务列表:"gsm8k_cot_zeroshot" "humaneval" "arc_challenge_chat" "gpqa_main_cot_n_shot" "ifeval" "mbpp"
declare -a TASKS=("gpqa_main_cot_n_shot")
# 模型列表
declare -a MODELS=(
    "../Model/OriginalModel/Qwen/Qwen3-0.6B"
    # "../Model/OriginalModel/Qwen/Qwen3-1.7B"
    # "../Model/OriginalModel/prithivMLmods/Theta-Crucis-0.6B-Turbo1"
    # "../Model/OriginalModel/prithivMLmods/Nenque-MoT-0.6B-Elite14"
    # "../Model/OriginalModel/prithivMLmods/Eta-Aurigae-0.6B-Echelon1"
    # "../Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k"
    # "../Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k_rw"
    # "../Model/MergeModel/Qwen3-0.6B-Mot_mix-sft-16.2k_dwi"
    # "../Model/MergeModel/Qwen3-0.6B-Mot_mixed-sft-194.4k"
)
OUTPUT_DIR="./eval_result/harness_results"
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


        # 执行任务并捕获退出状态，添加 --output-dir 参数
        if accelerate launch -m lm_eval --model hf --model_args pretrained="$model" --tasks "$task" --device cuda:0,1,2,3 --batch_size 1 --output_path "$OUTPUT_DIR" --confirm_run_unsafe_code --limit 150 ; then
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