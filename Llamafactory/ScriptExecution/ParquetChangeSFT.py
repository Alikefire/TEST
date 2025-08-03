import pyarrow.parquet as pq
import json
from collections import defaultdict
from tqdm import tqdm
import os
import glob
from concurrent.futures import ThreadPoolExecutor

def validate_sharegpt(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)

    error_log = defaultdict(int)

    for entry in data:
        conversations = entry["conversations"]
        for i, msg in enumerate(conversations):
            # 检查角色位置规则
            if (i % 2 == 0) and (msg["from"] != "human"):
                error_log["human_position_error"] += 1
            elif (i % 2 == 1) and (msg["from"] != "gpt"):
                error_log["gpt_position_error"] += 1

            # 检查消息内容非空
            if not msg["value"].strip():
                error_log["empty_content"] += 1

    print("Validation Report:")
    for err, count in error_log.items():
        print(f"{err}: {count} errors")

def convert_parquet_to_sharegpt(parquet_path, output_path):
    """
    Parquet转ShareGPT格式转换器
    只处理messages字段，严格遵守奇偶位置规则
    """
    # 读取Parquet文件
    table = pq.read_table(
        parquet_path,
        memory_map=True,
        use_threads=True,
        buffer_size=1024 * 1024,   # 1MB
    )
    df = table.to_pandas()

    output_data = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        conversations = []
        prev_role = None

        # 提取question和deepseek_thinking_trajectory列
        question = row.get('question', '')
        thinking_trajectory = row.get('deepseek_thinking_trajectory', '')
        
        # 确保数据不为空
        if not question or not thinking_trajectory:
            error_log.append(f"行 {idx} 缺少必要数据: question={bool(question)}, deepseek_thinking_trajectory={bool(thinking_trajectory)}")
            continue


        # 添加当前消息
        conversations.append({
            "from": 'human',
            "value": question
        })
        conversations.append({
            "from": 'gpt',
            "value": thinking_trajectory
        })


        # 构建最终对象
        output_entry = {
            "conversations": conversations,
            "system": "",  # 保持为空
            "tools": ""  # 保持为空
        }

        output_data.append(output_entry)

    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)



def batch_convert(input_dir, output_dir, max_workers=4):
    """
    批量转换Parquet文件为ShareGPT格式

    参数：
    input_dir: 包含Parquet文件的输入目录
    output_dir: 输出JSON文件的目录
    max_workers: 并行工作线程数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有Parquet文件（匹配您的命名规范）
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "train-*-of-*.parquet")))
    print(input_dir)
    # 从第 376 个文件开始处理
    start_index = 375
    # 创建处理进度条
    pbar = tqdm(total=len(parquet_files), desc="Processing Files")

    # 错误日志记录器
    error_log = []

    def process_file(file_path):
        try:
            # 生成输出路径
            base_name = os.path.basename(file_path).replace(".parquet", ".json")
            output_path = os.path.join(output_dir, base_name)
            print("convertion fail")
            # 调用现有转换函数
            convert_parquet_to_sharegpt(file_path, output_path)

            pbar.update(1)
            return True
        except Exception as e:
            error_log.append(f"Error processing {file_path}: {str(e)}")
            pbar.update(1)
            return False

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, fp) for fp in parquet_files[start_index:]]# 从第 start_index 个文件开始处理

        # 等待所有任务完成
        for future in futures:
            future.result()

    pbar.close()

    # 保存错误日志
    if error_log:
        error_path = os.path.join(output_dir, "conversion_errors.log")
        with open(error_path, "w") as f:
            f.write("\n".join(error_log))
        print(f"完成转换，遇到 {len(error_log)} 个错误，详见 {error_path}")
    else:
        print("所有文件转换成功！")

#批量处理
# if __name__ == "__main__":
#     # 使用示例（根据硬件配置调整线程数）
#     batch_convert(
#         input_dir="/home/zdd/xx_help/LLaMA-Factory/data/axolotl-ai-co/numina-cot-logprobs-859k-8b-sft/data",  # Parquet文件所在目录
#         output_dir="/home/zdd/xx_help/LLaMA-Factory/data/axolotl-ai-co/numina-cot-logprobs-859k-8b-sft/data_json",  # JSON输出目录
#         max_workers=8  # 推荐设置为CPU核心数×2
#     )
#单个文件处理
if __name__ == "__main__":
    # 使用示例
    convert_parquet_to_sharegpt(
        parquet_path="/home/xiexin/xx_help/LLaMA-Factory/data/simplescaling/s1K-1.1/data/train-00000-of-00001.parquet",
        output_path="/home/xiexin/xx_help/LLaMA-Factory/data/simplescaling/s1K-1.1/data/train.json"
    )