import pandas as pd
import json

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

# 示例用法
if __name__ == "__main__":
    # 假设您的Parquet文件名为 'input.parquet'
    # 假设您希望输出的JSONL文件名为 'output.jsonl'
    subset_name="math"
    max_samples_to_read = 200  # 设置您希望读取的最大样本数目
    input_file = f"./data/open-r1/Mixture-of-Thoughts/{subset_name}/train-00001-of-00004.parquet"
    output_file = f"./data/open-r1/Mixture-of-Thoughts/mix_train_data/{subset_name}/{subset_name}_{max_samples_to_read}.jsonl"


    # 运行转换
    convert_parquet_to_jsonl(input_file, output_file, max_samples=max_samples_to_read)