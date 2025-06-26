import pyarrow.parquet as pq
import json
from collections import defaultdict
from tqdm import tqdm
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import re

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
    新版逻辑：
    1. 移除奇偶校验规则
    2. chosen来自messages中最后一个assistant内容
    3. rejected来自generations列第一个元素的第二个双引号内容
    """
    # 读取Parquet文件
    table = pq.read_table(
        parquet_path,
        memory_map=True,
        use_threads=True,
        buffer_size=1024 * 1024,
    )
    df = table.to_pandas()

    output_data = []
    error_log = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
       try:
            conversations = []
            last_assistant_content = "[未找到有效回答]"

            # 处理messages字段
            for msg in row['messages']:
                current_role = msg['role'].lower()
                content = msg['content'].strip()

                # 角色转换
                if current_role == 'user':
                    role = 'human'
                    conversations.append({
                        "from": role,
                        "value": content
                    })
                elif current_role == 'assistant':
                    role = 'gpt'
                    last_assistant_content = content  # 记录最后一个assistant回复
                else:
                    continue


            # 处理generations字段
            #根据实际的数据结构来调整提取逻辑。如果generations字段存储的是列表的列表，那么应该直接访问，而不是用正则表达式。
            # 如果存储的是字符串形式的列表，如'["a", "b"]'，则需要用正则表达式提取。
            # if row.get('generations'):
            #     first_generation = row['generations']
            # # 尝试直接访问列表元素
            # if isinstance(first_generation, list):
            #     if len(first_generation) >= 2:
            #         rejected_value = first_generation[1]

            # print(extract_second_quoted('["<think>你好","<think>你也好"]'))#测试extract_second_quoted函数有效性
            def extract_second_quoted(text):
                matches = re.findall(r'"([^"]*)"', str(text))
                return matches[1] if len(matches) >= 2 else ""

            # 初始化默认拒绝值
            rejected_value = ""

            # # 从generations字段中提取有效值：取第一个生成结果中的第二个双引号内容
            if len(row['generations']) > 0:
                first_generation = row['generations']
                rejected_value = extract_second_quoted(first_generation)
            #     # print(first_generation)

            # 构建最终对象
            output_entry = {
                "conversations": conversations,
                "chosen": {
                    "from": "gpt",
                    "value": last_assistant_content
                },
                "rejected": {
                    "from": "gpt",
                    "value": rejected_value
                }
            }
            output_data.append(output_entry)
        #except语句需要在最后一行
       except Exception as e:
           error_log.append(f"Row {idx} 处理失败: {str(e)}")
           continue
    try:
        # 保存结果,在文件写入阶段，如果没有异常处理，则文件即使没有完全写入也没有报错信息
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        # 检查最后一个条目
        with open(output_path) as f:
            data = json.load(f)
            print("最后条目结构:", json.dumps(data[-1], indent=2))
    except Exception as e:
        print(f"保存文件时发生错误: {e}")

    if error_log:
        print("\n错误日志（前10条）：")
        print('\n'.join(error_log[:10]))

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
    # 从第start_index个文件开始处理
    start_index = 2
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
if __name__ == "__main__":
    # 使用示例（根据硬件配置调整线程数）
    batch_convert(
        input_dir="/home/zdd/xx_help/LLaMA-Factory/data/open-r1/OpenR1-Math-220k/data",  # Parquet文件所在目录
        output_dir="/home/zdd/xx_help/LLaMA-Factory/data/open-r1/OpenR1-Math-220k/data_dpo_json",  # JSON输出目录
        max_workers=8  # 推荐设置为CPU核心数×2
    )
#单个文件处理
# if __name__ == "__main__":
#     # 使用示例
#     convert_parquet_to_sharegpt(
#         parquet_path="/home/zdd/xx_help/LLaMA-Factory/data/open-r1/OpenR1-Math-220k/data/train-00000-of-00010.parquet",
#         output_path="/home/zdd/xx_help/LLaMA-Factory/data/open-r1/OpenR1-Math-220k/data/train-00000-of-00010.json"
#     )