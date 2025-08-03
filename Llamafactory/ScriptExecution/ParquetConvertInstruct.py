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
    将特定格式的Parquet文件转换为JSONL文件。

    Parquet文件格式示例：
    每一行是一个字典，包含一个键 "messages"，其值是一个对话列表：
    {"messages": [
      {"content": "user_content_1", "role": "user"},
      {"content": "assistant_content_1", "role": "assistant"},
      {"content": "user_content_2", "role": "user"},
      {"content": "assistant_content_2", "role": "assistant"}
    ]}

    JSONL文件格式示例：
    {"instruction": "user_content_1", "input": "", "output": "assistant_content_1"}
    {"instruction": "user_content_2", "input": "", "output": "assistant_content_2"}
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
        # 假设每一行数据是一个包含'messages'键的字典，其值是对话列表
        dialogues = row.get('messages') # 获取'messages'键的值

        # 检查dialogues是否是列表且包含至少两个元素（用户和助手）
        #使用if not isinstance(dialogues, list)进行list检查有误，可能是parquet文件的独特格式
        if len(dialogues) < 2:
            print(f"Skipping row {index} due to invalid dialogue format: {row}")
            continue

        # 遍历对话，每两项（用户和助手）组成一个instruction-output对
        for i in range(0, len(dialogues), 2):
            if i + 1 < len(dialogues):
                user_message = dialogues[i]
                assistant_message = dialogues[i+1]

                if user_message.get('role') == 'user' and assistant_message.get('role') == 'assistant':
                    output_data.append({
                        "instruction": user_message.get('content', ''),
                        "input": "",
                        "output": assistant_message.get('content', '')
                    })
                else:
                    print(f"Warning: Unexpected role sequence at row {index}, dialogue part {i}. Expected 'user' then 'assistant'. Got: {user_message.get('role')}, {assistant_message.get('role')}")
            else:
                print(f"Warning: Incomplete dialogue pair at row {index}, starting at part {i}. Skipping last message.")

    # 将转换后的数据写入JSONL文件
    with open(output_jsonl_path, 'w', encoding='utf-8') as f:
        for entry in output_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Conversion complete. Converted {len(output_data)} entries to {output_jsonl_path}")

def convert_parquet_to_sharegpt(input_parquet_path, output_jsonl_path, max_samples=None):
    """
    将特定格式的Parquet文件转换为ShareGPT格式的JSONL文件。

    Parquet文件格式示例：
    每一行是一个字典，包含一个键 "messages"，其值是一个对话列表：
    {"messages": [
      {"content": "user_content_1", "role": "user"},
      {"content": "assistant_content_1", "role": "assistant"},
      {"content": "user_content_2", "role": "user"},
      {"content": "assistant_content_2", "role": "assistant"}
    ]}

    期望的ShareGPT格式示例 (每个用户-助手对一个条目):
    {
      "conversations": [
        {"from": "human", "value": "人类指令"},
        {"from": "gpt", "value": "模型回答"}
      ],
      "system": " ",
      "tools": " "
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
        # 假设每一行数据是一个包含'messages'键的字典，其值是对话列表
        dialogues = row.get('messages') # 获取'messages'键的值

        # 检查dialogues是否是列表且包含至少两个元素（用户和助手）
        if len(dialogues) < 2:
            print(f"Skipping row {index} due to invalid dialogue format: {row}")
            continue

        # 转换为ShareGPT格式的对话
        conversations = []
        for dialogue in dialogues:
            if dialogue.get('role') == 'user':
                conversations.append({
                    "from": "human",
                    "value": dialogue.get('content', '')
                })
            elif dialogue.get('role') == 'assistant':
                conversations.append({
                    "from": "gpt",
                    "value": dialogue.get('content', '')
                })
            else:
                print(f"Warning: Unknown role '{dialogue.get('role')}' at row {index}. Skipping this message.")

        # 只有当conversations不为空时才添加到输出数据
        if conversations:
            output_data.append({
                "conversations": conversations,
                "system": "",
                "tools": ""
            })

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

def batch_convert(input_dir, output_dir, max_workers=4, max_samples=None):
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
    # 从第 start_index 个文件开始处理
    start_index = 0
    # 创建处理进度条
    pbar = tqdm(total=len(parquet_files), desc="Processing Files")

    # 错误日志记录器
    error_log = []

    def process_file(file_path):
        try:
            # 生成输出路径
            base_name = os.path.basename(file_path).replace(".parquet", ".jsonl")
            output_path = os.path.join(output_dir, base_name)
            print("convertion starts")
            # 调用现有转换函数
            # convert_parquet_with_format(file_path, output_path, "instruct",max_samples=max_samples)
            convert_parquet_with_format(file_path, output_path, "sharegpt",max_samples=max_samples)

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
# 示例用法
if __name__ == "__main__":
    # 假设您的Parquet文件名为 'input.parquet'
    # 假设您希望输出的JSONL文件名为 'output.jsonl'
    subset_names=["code", "math", "science"]
    max_samples_to_read = 250  # 设置您希望读取的每个文件的最大样本数目
    for subset_name in subset_names:
        input_dir = f"/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/{subset_name}"
        # # 生成instruct和sharegpt格式
        output_dir = f"/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/mix_train_data/sharegpt_4k/{subset_name}"
        batch_convert(input_dir, output_dir, max_workers=4, max_samples=max_samples_to_read)