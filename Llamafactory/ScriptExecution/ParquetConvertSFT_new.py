import pyarrow.parquet as pq
import json
from collections import defaultdict
from tqdm import tqdm
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import re

def convert_parquet_to_jsonl(input_parquet_path, output_jsonl_path, max_samples=None):
    """
    将包含conversations字段的Parquet文件转换为Instruct格式的JSONL文件。

    Parquet文件格式示例：
    每一行是一个字典，包含一个键 "conversations"，其值是一个对话列表：
    {"conversations": [
      {"from": "human", "value": "用户问题"},
      {"from": "gpt", "value": "GPT回答"}
    ]}

    JSONL文件格式示例：
    {"instruction": "用户问题", "input": "", "output": "GPT回答"}
    """
    try:
        # 读取Parquet文件
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    output_data = []
    # 如果设置了max_samples，则只处理前max_samples行
    if max_samples is not None:
        df = df.head(max_samples)

    for index, row in df.iterrows():
        try:
            # 获取conversations字段
            conversations = row.get('conversations', [])
            
            # 转换为Python原生类型，避免numpy数组导致的JSON序列化错误
            if hasattr(conversations, 'tolist'):
                conversations = conversations.tolist()
            elif not isinstance(conversations, list):
                conversations = list(conversations) if conversations is not None else []

            # 检查conversations是否是列表且包含至少两个元素（用户和助手）
            if not conversations or len(conversations) < 2:
                print(f"Skipping row {index} due to invalid conversations format")
                continue

            # 遍历对话，每两项（human和gpt）组成一个instruction-output对
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    human_message = conversations[i]
                    gpt_message = conversations[i+1]

                    # 确保消息是字典类型
                    if isinstance(human_message, dict) and isinstance(gpt_message, dict):
                        if human_message.get('from') == 'human' and gpt_message.get('from') == 'gpt':
                            output_data.append({
                                "instruction": str(human_message.get('value', '')),
                                "input": "",
                                "output": str(gpt_message.get('value', ''))
                            })
                        else:
                            print(f"Warning: Unexpected role sequence at row {index}, dialogue part {i}. Expected 'human' then 'gpt'. Got: {human_message.get('from')}, {gpt_message.get('from')}")
                    else:
                        print(f"Warning: Invalid message format at row {index}, dialogue part {i}")
                else:
                    print(f"Warning: Incomplete dialogue pair at row {index}, starting at part {i}. Skipping last message.")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # 将转换后的数据写入JSONL文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Conversion complete. Converted {len(output_data)} entries to {output_jsonl_path}")

def convert_parquet_to_sharegpt(input_parquet_path, output_jsonl_path, max_samples=None):
    """
    将包含conversations字段的Parquet文件转换为ShareGPT格式的JSONL文件。

    Parquet文件格式示例：
    每一行是一个字典，包含一个键 "conversations"，其值是一个对话列表：
    {"conversations": [
      {"from": "human", "value": "用户问题"},
      {"from": "gpt", "value": "GPT回答"}
    ]}

    期望的ShareGPT格式示例:
    {
      "conversations": [
        {"from": "human", "value": "人类指令"},
        {"from": "gpt", "value": "模型回答"}
      ],
      "system": "",
      "tools": ""
    }
    """
    try:
        # 读取Parquet文件
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    output_data = []
    # 如果设置了max_samples，则只处理前max_samples行
    if max_samples is not None:
        df = df.head(max_samples)

    for index, row in df.iterrows():
        try:
            # 获取conversations字段
            conversations = row.get('conversations', [])
            
            # 转换为Python原生类型，避免numpy数组导致的JSON序列化错误
            if hasattr(conversations, 'tolist'):
                conversations = conversations.tolist()
            elif not isinstance(conversations, list):
                conversations = list(conversations) if conversations is not None else []

            # 检查conversations是否是列表且包含至少一个元素
            if not conversations:
                print(f"Skipping row {index} due to empty conversations")
                continue

            # 确保conversations中的每个消息都是Python原生类型
            processed_conversations = []
            for msg in conversations:
                if isinstance(msg, dict):
                    processed_msg = {
                        "from": str(msg.get('from', '')),
                        "value": str(msg.get('value', ''))
                    }
                    processed_conversations.append(processed_msg)
                else:
                    print(f"Warning: Invalid message format at row {index}: {msg}")

            # 只有当processed_conversations不为空时才添加到输出数据
            if processed_conversations:
                output_data.append({
                    "conversations": processed_conversations,
                    "system": "",
                    "tools": ""
                })
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue

    # 将转换后的数据写入JSONL文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Conversion complete. Converted {len(output_data)} entries to {output_jsonl_path}")

def convert_parquet_with_format(input_parquet_path, output_jsonl_path, format_type="instruct", max_samples=None):
    """
    根据指定格式将Parquet文件转换为JSONL文件。
    
    Args:
        input_parquet_path: 输入的Parquet文件路径
        output_jsonl_path: 输出的JSONL文件路径
        format_type: 输出格式类型，"instruct" 或 "sharegpt"
        max_samples: 最大处理样本数
    """
    if format_type == "instruct":
        convert_parquet_to_jsonl(input_parquet_path, output_jsonl_path, max_samples)
    elif format_type == "sharegpt":
        convert_parquet_to_sharegpt(input_parquet_path, output_jsonl_path, max_samples)
    else:
        print(f"Error: Unsupported format type '{format_type}'. Supported formats: 'instruct', 'sharegpt'")

def batch_convert(input_dir, output_dir, format_type="both", max_workers=4, max_samples=None):
    """
    批量转换Parquet文件

    参数：
    input_dir: 包含Parquet文件的输入目录
    output_dir: 输出JSON文件的目录
    format_type: 输出格式类型，"instruct", "sharegpt" 或 "both"
    max_workers: 并行工作线程数
    max_samples: 每个文件的最大处理样本数
    """
    # 创建输出目录
    if format_type in ["instruct", "both"]:
        os.makedirs(os.path.join(output_dir, "instruct"), exist_ok=True)
    if format_type in ["sharegpt", "both"]:
        os.makedirs(os.path.join(output_dir, "sharegpt"), exist_ok=True)

    # 获取所有Parquet文件
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "*.parquet")))
    if not parquet_files:
        parquet_files = sorted(glob.glob(os.path.join(input_dir, "**/*.parquet"), recursive=True))
    
    print(f"Found {len(parquet_files)} parquet files in {input_dir}")
    
    if not parquet_files:
        print("No parquet files found!")
        return
    
    # 创建处理进度条
    pbar = tqdm(total=len(parquet_files), desc="Processing Files")
    
    # 错误日志记录器
    error_log = []

    def process_file(file_path):
        try:
            base_name = os.path.basename(file_path).replace(".parquet", ".jsonl")
            print(f"Processing: {file_path}")
            
            # 根据format_type生成不同格式的文件
            if format_type in ["instruct", "both"]:
                instruct_output_path = os.path.join(output_dir, "instruct", base_name)
                convert_parquet_with_format(file_path, instruct_output_path, "instruct", max_samples)
            
            if format_type in ["sharegpt", "both"]:
                sharegpt_output_path = os.path.join(output_dir, "sharegpt", base_name)
                convert_parquet_with_format(file_path, sharegpt_output_path, "sharegpt", max_samples)
            
            pbar.update(1)
            return True
        except Exception as e:
            error_log.append(f"Error processing {file_path}: {str(e)}")
            pbar.update(1)
            return False

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, fp) for fp in parquet_files]
        
        # 等待所有任务完成
        for future in futures:
            future.result()

    pbar.close()

    # 保存错误日志
    if error_log:
        error_path = os.path.join(output_dir, "conversion_errors.log")
        with open(error_path, "w", encoding='utf-8') as f:
            f.write("\n".join(error_log))
        print(f"完成转换，遇到 {len(error_log)} 个错误，详见 {error_path}")
    else:
        print("所有文件转换成功！")

# 示例用法
if __name__ == "__main__":
    # 配置参数
    input_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhaoziyu-240108120122/xx_help/LLaMA-Factory/data/mlabonne/FineTome-100k/data"  # 替换为您的输入目录
    output_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhaoziyu-240108120122/xx_help/LLaMA-Factory/data/mlabonne/FineTome-100k/data"  # 替换为您的输出目录
    max_samples_per_file = 3000  # 每个文件最大处理样本数，None表示处理全部
    
    # 执行批量转换
    batch_convert(
        input_dir=input_dir,
        output_dir=output_dir,
        format_type="instruct",  # 生成instruct和sharegpt两种格式
        max_workers=4,
        max_samples=max_samples_per_file
    )