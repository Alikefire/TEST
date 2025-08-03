#!/bin/bash
# 批量checkpoint监控和评估脚本

export HF_ALLOW_CODE_EVAL="1"
# 配置参数
BASE_MODEL_DIR="../Model/MergeModel/Qwen3-0.6B-Mot_mixed-sft-194.4k"
CHECKPOINT_INTERVAL=900  # checkpoint保存间隔步数
WAIT_TIME=1  # 等待时间（秒）
START_CHECKPOINT=900  # 起始checkpoint编号
END_CHECKPOINT=8100   # 结束checkpoint编号

# 任务列表:"aime", "amc", "math", "minerva", "olympiad_bench","aime25","gpqa"
# 任务列表:"gsm8k_cot_zeroshot" "humaneval" "arc_challenge_chat" "gpqa_main_cot_n_shot" "ifeval" "mbpp"
declare -a TASKS=("humaneval")
OUTPUT_DIR="./eval_result/harness_results" #./eval_result/deepscaler_results

# 颜色定义
COLOR_GREEN='\033[1;32m'
COLOR_BLUE='\033[1;34m'
COLOR_YELLOW='\033[1;33m'
COLOR_RED='\033[1;31m'
COLOR_RESET='\033[0m'

# 记录已处理的checkpoint
PROCESSED_CHECKPOINTS_FILE="./processed_checkpoints.log"
touch "$PROCESSED_CHECKPOINTS_FILE"

# 函数：检查checkpoint是否已处理
check_if_processed() {
    local checkpoint_path="$1"
    grep -Fxq "$checkpoint_path" "$PROCESSED_CHECKPOINTS_FILE"
}

# 函数：标记checkpoint为已处理
mark_as_processed() {
    local checkpoint_path="$1"
    echo "$checkpoint_path" >> "$PROCESSED_CHECKPOINTS_FILE"
}

# 函数：执行评估
run_evaluation() {
    local model_path="$1"
    local checkpoint_num="$2"
    
    echo -e "\n${COLOR_GREEN}[$(date +'%T')] 开始评估模型: $model_path ${COLOR_RESET}"
    
    for task in "${TASKS[@]}"; do
        echo -e "\n${COLOR_BLUE}[$(date +'%T')] 开始执行任务: $task (checkpoint-$checkpoint_num) ${COLOR_RESET}"
        
        # 记录开始时间
        start_time=$(date +%s)
        RUN_TIMESTAMP=$(date +'%Y%m%d-%H%M')
        
        # 构建输出目录
        task_output_dir="${OUTPUT_DIR}/${task}/checkpoint-${checkpoint_num}/${RUN_TIMESTAMP}"
        mkdir -p "$task_output_dir"
        
        # 执行deepscaler评估
        # if ./Light-R1/deepscaler-release/scripts/eval/eval_model.sh --datasets "$task" --model "$model_path" --output-dir "$task_output_dir"; then
        # 执行lm-harness评估
        if accelerate launch -m lm_eval --model hf --model_args pretrained="$model_path" --tasks "$task" --device cuda:1 --batch_size 1 --output_path "$task_output_dir" --confirm_run_unsafe_code --limit 150 ; then
            end_time=$(date +%s)
            duration=$((end_time - start_time))
            echo -e "${COLOR_GREEN}✓ [$(date +'%T')] 任务 $task 已完成 (耗时: ${duration}s) ${COLOR_RESET}"
        else
            echo -e "${COLOR_RED}✗ [$(date +'%T')] 任务 $task 执行失败 ${COLOR_RESET}"
        fi
    done
    
    # 标记为已处理
    mark_as_processed "$model_path"
}

# 函数：获取checkpoint编号
get_checkpoint_number() {
    local checkpoint_path="$1"
    echo "$checkpoint_path" | grep -o 'checkpoint-[0-9]*' | grep -o '[0-9]*'
}

# 函数：检查checkpoint编号是否在范围内
is_checkpoint_in_range() {
    local checkpoint_num="$1"
    [ "$checkpoint_num" -ge "$START_CHECKPOINT" ] && [ "$checkpoint_num" -le "$END_CHECKPOINT" ]
}

# 函数：获取已处理的范围内checkpoint数量
get_processed_count_in_range() {
    local count=0
    for ((i=START_CHECKPOINT; i<=END_CHECKPOINT; i+=CHECKPOINT_INTERVAL)); do
        checkpoint_path="$BASE_MODEL_DIR/checkpoint-$i"
        if check_if_processed "$checkpoint_path"; then
            count=$((count + 1))
        fi
    done
    echo $count
}

# 主监控循环
echo -e "${COLOR_YELLOW}============ 开始监控checkpoint目录 ============${COLOR_RESET}"
echo -e "${COLOR_YELLOW}监控目录: $BASE_MODEL_DIR${COLOR_RESET}"
echo -e "${COLOR_YELLOW}checkpoint间隔: $CHECKPOINT_INTERVAL 步${COLOR_RESET}"
echo -e "${COLOR_YELLOW}等待时间: $WAIT_TIME 秒${COLOR_RESET}"
echo -e "${COLOR_YELLOW}起始checkpoint: $START_CHECKPOINT${COLOR_RESET}"
echo -e "${COLOR_YELLOW}结束checkpoint: $END_CHECKPOINT${COLOR_RESET}\n"

# 计算总的checkpoint数量
total_checkpoints=$(( (END_CHECKPOINT - START_CHECKPOINT) / CHECKPOINT_INTERVAL + 1 ))
echo -e "${COLOR_BLUE}预期处理 $total_checkpoints 个checkpoint${COLOR_RESET}\n"

while true; do
    # 查找所有checkpoint目录
    checkpoints=()
    while IFS= read -r -d '' checkpoint; do
        checkpoints+=("$checkpoint")
    done < <(find "$BASE_MODEL_DIR" -maxdepth 1 -type d -name "checkpoint-*" -print0 2>/dev/null | sort -z)
    
    # 检查是否有新的checkpoint
    new_checkpoint_found=false
    
    for checkpoint_path in "${checkpoints[@]}"; do
        if ! check_if_processed "$checkpoint_path"; then
            checkpoint_num=$(get_checkpoint_number "$checkpoint_path")
            
            # 检查checkpoint是否在指定范围内
            if is_checkpoint_in_range "$checkpoint_num"; then
                echo -e "${COLOR_YELLOW}[$(date +'%T')] 发现新checkpoint: $checkpoint_path (编号: $checkpoint_num)${COLOR_RESET}"
                
                # 等待指定时间
                echo -e "${COLOR_BLUE}[$(date +'%T')] 等待 $WAIT_TIME 秒后开始评估...${COLOR_RESET}"
                sleep $WAIT_TIME
                
                # 执行评估
                run_evaluation "$checkpoint_path" "$checkpoint_num"
                
                new_checkpoint_found=true
                
                # 获取当前已处理的数量
                processed_count=$(get_processed_count_in_range)
                echo -e "${COLOR_GREEN}[$(date +'%T')] 已处理 $processed_count/$total_checkpoints 个checkpoint (范围内)${COLOR_RESET}\n"
                break
            else
                echo -e "${COLOR_YELLOW}[$(date +'%T')] 跳过checkpoint: $checkpoint_path (编号: $checkpoint_num, 不在范围内)${COLOR_RESET}"
                # 标记为已处理，避免重复检查
                mark_as_processed "$checkpoint_path"
            fi
        fi
    done
    
    # 检查是否已处理完所有范围内的checkpoint
    processed_count=$(get_processed_count_in_range)
    if [ "$processed_count" -eq "$total_checkpoints" ]; then
        echo -e "${COLOR_GREEN}[$(date +'%T')] 所有范围内的checkpoint已处理完毕${COLOR_RESET}"
        break
    fi
    
    # 如果没有找到新checkpoint，等待一段时间再检查
    if [ "$new_checkpoint_found" = false ]; then
        echo -e "${COLOR_BLUE}[$(date +'%T')] 等待新checkpoint生成... (已处理: $processed_count/$total_checkpoints)${COLOR_RESET}"
        sleep 30  # 每30秒检查一次
    fi
done

echo -e "\n${COLOR_GREEN}============ 所有checkpoint评估完毕 ============${COLOR_RESET}"
echo -e "${COLOR_GREEN}共处理了 $processed_count 个checkpoint (范围: $START_CHECKPOINT - $END_CHECKPOINT)${COLOR_RESET}"