import pyarrow.parquet as pq
import json
from collections import defaultdict
from tqdm import tqdm
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import re
import tiktoken
import gc
import threading
from typing import Dict, List
from transformers import AutoTokenizer

# 线程本地存储tokenizer，避免全局共享
thread_local_data = threading.local()

# Qwen3-0.6B tokenizer路径
TOKENIZER_PATH = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhaoziyu-240108120122/xx_help/LLaMA-Factory/Model/OriginalModel/Qwen/Qwen3-0.6B"

def get_tokenizer():
    """获取线程本地的Qwen3-0.6B tokenizer"""
    if not hasattr(thread_local_data, 'tokenizer'):
        try:
            # 使用本地tokenizer文件，完全离线
            thread_local_data.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_PATH, 
                local_files_only=True,
                trust_remote_code=True  # Qwen模型需要这个参数
            )
            print(f"Successfully loaded Qwen3-0.6B tokenizer from {TOKENIZER_PATH}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            print("Falling back to character-based estimation")
            thread_local_data.tokenizer = None
    return thread_local_data.tokenizer

def count_tokens(text):
    """计算文本的token数量（使用Qwen3-0.6B tokenizer）"""
    if not text:
        return 0
    
    tokenizer = get_tokenizer()
    if tokenizer is not None:
        try:
            # 使用Qwen3-0.6B tokenizer进行精确计算
            tokens = tokenizer.encode(str(text), add_special_tokens=False)
            return len(tokens)
        except Exception as e:
            print(f"Error tokenizing text: {e}")
            # 如果tokenizer出错，回退到估算方法
            return count_tokens_fallback(text)
    else:
        # 如果tokenizer加载失败，使用估算方法
        return count_tokens_fallback(text)

def count_tokens_fallback(text):
    """备用的token数量估算方法"""
    if not text:
        return 0
    
    text = str(text)
    
    # 统计中英文字符
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    other_chars = len(text) - chinese_chars - english_chars
    
    # 估算token数量（针对Qwen模型调整）
    # 中文：每个字符约1.5个tokens（Qwen对中文优化较好）
    # 英文：每4个字符约1个token
    # 其他字符：每个字符约1个token
    estimated_tokens = int(
        chinese_chars * 1.5 + 
        english_chars * 0.25 + 
        other_chars * 1.0
    )
    
    return max(1, estimated_tokens)  # 至少返回1

def get_token_range(token_count):
    """根据token数量确定范围目录"""
    if token_count <= 6000:
        return "0-6k"
    elif token_count <= 12000:
        return "6k-12k"
    elif token_count <= 18000:
        return "12k-18k"
    else:
        return "18k+"

def get_difficulty_dir(difficulty, domain):
    """获取difficulty目录名（仅对code领域）"""
    if domain.lower() == "code" and difficulty in ["6", "7", "8", "9", "10"]:
        return f"difficulty_{difficulty}"
    return None

def convert_parquet_to_instruct(input_parquet_path, base_output_dir, max_samples=None):
    """
    将Parquet文件转换为Instruct格式的JSONL文件，按domain、token长度、difficulty分类
    
    Parquet格式：
    {
        "difficulty": "7",
        "source": "某来源",
        "domain": "code",
        "conversations": [
            {"from": "human", "value": "用户问题"},
            {"from": "gpt", "value": "GPT回答"}
        ]
    }
    
    Instruct格式：
    {"instruction": "用户问题", "input": "", "output": "GPT回答"}
    """
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    if max_samples is not None:
        df = df.head(max_samples)

    # 按分类组织数据
    categorized_data = defaultdict(list)
    
    for index, row in df.iterrows():
        try:
            difficulty = str(row.get('difficulty', '')).strip()
            source = row.get('source', '')
            domain = str(row.get('domain', '')).strip().lower()
            conversations = row.get('conversations', [])
            
            # 转换为Python原生类型，避免numpy数组导致的JSON序列化错误
            if hasattr(conversations, 'tolist'):
                conversations = conversations.tolist()
            elif not isinstance(conversations, list):
                conversations = list(conversations) if conversations is not None else []
            
            # 验证domain是否在允许的范围内
            if domain not in ['code', 'math', 'science']:
                print(f"Warning: Unknown domain '{domain}' at row {index}. Skipping.")
                continue
                
            if conversations is None or len(conversations) == 0 or len(conversations) < 2:
                print(f"Skipping row {index} due to invalid conversations format")
                continue
            
            # 处理对话对
            for i in range(0, len(conversations), 2):
                if i + 1 < len(conversations):
                    human_msg = conversations[i]
                    gpt_msg = conversations[i + 1]
                    
                    # 确保消息内容是Python原生类型
                    if isinstance(human_msg, dict) and isinstance(gpt_msg, dict):
                        if (human_msg.get('from') == 'human' and 
                            gpt_msg.get('from') == 'gpt'):
                            
                            gpt_content = str(gpt_msg.get('value', ''))
                            human_content = str(human_msg.get('value', ''))
                            
                            # 计算GPT回复的token长度（使用Qwen3-0.6B tokenizer）
                            token_count = count_tokens(gpt_content)
                            token_range = get_token_range(token_count)
                            
                            # 构建目录路径
                            path_parts = [domain, token_range]
                            
                            # 如果是code领域，添加difficulty分类
                            if domain == 'code':
                                difficulty_dir = get_difficulty_dir(difficulty, domain)
                                if difficulty_dir:
                                    path_parts.append(difficulty_dir)
                                else:
                                    print(f"Warning: Invalid difficulty '{difficulty}' for code domain at row {index}")
                                    continue
                            
                            category_key = '/'.join(path_parts)
                            
                            entry = {
                                "instruction": human_content,
                                "input": "",
                                "output": gpt_content
                            }
                            
                            categorized_data[category_key].append(entry)
                        
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    # 显式释放内存
    del df
    gc.collect()
    print(f"Memory released for {input_parquet_path}")

    # 写入分类文件
    total_entries = 0
    for category, entries in categorized_data.items():
        if not entries:
            continue
            
        # 创建目录结构
        output_dir = os.path.join(base_output_dir, 'instruct', category)
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        base_name = os.path.basename(input_parquet_path).replace('.parquet', '.jsonl')
        output_path = os.path.join(output_dir, base_name)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        total_entries += len(entries)
        print(f"Instruct format: Saved {len(entries)} entries to {output_path}")
    
    # 最终内存清理
    del categorized_data
    gc.collect()
    
    # 清理线程本地tokenizer
    if hasattr(thread_local_data, 'tokenizer'):
        del thread_local_data.tokenizer
    

    print(f"Instruct conversion complete. Total entries: {total_entries}")

def convert_parquet_to_sharegpt(input_parquet_path, base_output_dir, max_samples=None):
    """
    将Parquet文件转换为ShareGPT格式的JSONL文件，按domain、token长度、difficulty分类
    
    ShareGPT格式：
    {
        "conversations": [
            {"from": "human", "value": "用户问题"},
            {"from": "gpt", "value": "GPT回答"}
        ],
        "system": "",
        "tools": ""
    }
    """
    try:
        df = pd.read_parquet(input_parquet_path)
    except Exception as e:
        print(f"Error reading Parquet file: {e}")
        return

    if max_samples is not None:
        df = df.head(max_samples)

    # 按分类组织数据
    categorized_data = defaultdict(list)
    
    for index, row in df.iterrows():
        try:
            difficulty = str(row.get('difficulty', '')).strip()
            source = row.get('source', '')
            domain = str(row.get('domain', '')).strip().lower()
            conversations = row.get('conversations', [])
            
            # 转换为Python原生类型，避免numpy数组导致的JSON序列化错误
            if hasattr(conversations, 'tolist'):
                conversations = conversations.tolist()
            elif not isinstance(conversations, list):
                conversations = list(conversations) if conversations is not None else []
            
            # 验证domain
            if domain not in ['code', 'math', 'science']:
                print(f"Warning: Unknown domain '{domain}' at row {index}. Skipping.")
                continue
                
            if conversations is None or len(conversations) == 0:
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
            
            # 计算所有GPT回复的总token长度
            total_gpt_tokens = 0
            for msg in processed_conversations:
                if msg.get('from') == 'gpt':
                    total_gpt_tokens += count_tokens(msg.get('value', ''))
            
            token_range = get_token_range(total_gpt_tokens)
            
            # 构建目录路径
            path_parts = [domain, token_range]
            
            # 如果是code领域，添加difficulty分类
            if domain == 'code':
                difficulty_dir = get_difficulty_dir(difficulty, domain)
                if difficulty_dir:
                    path_parts.append(difficulty_dir)
                else:
                    print(f"Warning: Invalid difficulty '{difficulty}' for code domain at row {index}")
                    continue
            
            category_key = '/'.join(path_parts)
            
            entry = {
                "conversations": processed_conversations,
                "system": "",
                "tools": ""
            }
            
            categorized_data[category_key].append(entry)
                        
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            continue
    
    # 显式释放内存
    del df
    gc.collect()
    print(f"Memory released for {input_parquet_path}")

    # 写入分类文件
    total_entries = 0
    for category, entries in categorized_data.items():
        if not entries:
            continue
            
        # 创建目录结构
        output_dir = os.path.join(base_output_dir, 'sharegpt', category)
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        base_name = os.path.basename(input_parquet_path).replace('.parquet', '.jsonl')
        output_path = os.path.join(output_dir, base_name)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        total_entries += len(entries)
        print(f"ShareGPT format: Saved {len(entries)} entries to {output_path}")
    
    print(f"ShareGPT conversion complete. Total entries: {total_entries}")

def convert_parquet_with_format(input_parquet_path, base_output_dir, format_type="both", max_samples=None):
    """
    根据指定格式将Parquet文件转换为JSONL文件
    
    Args:
        input_parquet_path: 输入的Parquet文件路径
        base_output_dir: 基础输出目录
        format_type: 输出格式类型，"instruct", "sharegpt" 或 "both"
        max_samples: 最大处理样本数
    """
    if format_type in ["instruct", "both"]:
        convert_parquet_to_instruct(input_parquet_path, base_output_dir, max_samples)
    
    if format_type in ["sharegpt", "both"]:
        convert_parquet_to_sharegpt(input_parquet_path, base_output_dir, max_samples)
    
    if format_type not in ["instruct", "sharegpt", "both"]:
        print(f"Error: Unsupported format type '{format_type}'. Supported formats: 'instruct', 'sharegpt', 'both'")

def batch_convert(input_dir, base_output_dir, format_type="both", max_workers=1, max_samples=None):
    """
    批量转换Parquet文件
    
    参数：
    input_dir: 包含Parquet文件的输入目录
    base_output_dir: 基础输出目录
    format_type: 输出格式类型
    max_workers: 并行工作线程数
    max_samples: 每个文件的最大处理样本数
    """
    # 创建基础输出目录
    os.makedirs(base_output_dir, exist_ok=True)

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
            print(f"Processing: {file_path}")
            convert_parquet_with_format(file_path, base_output_dir, format_type, max_samples)
            pbar.update(1)
            return True
        except Exception as e:
            error_log.append(f"Error processing {file_path}: {str(e)}")
            pbar.update(1)
            return False

    # 使用线程池并行处理（建议max_workers=1避免内存问题）
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, fp) for fp in parquet_files]
        
        # 等待所有任务完成
        for future in futures:
            future.result()

    pbar.close()

    # 保存错误日志
    if error_log:
        error_path = os.path.join(base_output_dir, "conversion_errors.log")
        with open(error_path, "w", encoding='utf-8') as f:
            f.write("\n".join(error_log))
        print(f"完成转换，遇到 {len(error_log)} 个错误，详见 {error_path}")
    else:
        print("所有文件转换成功！")

# 示例用法
if __name__ == "__main__":
    # 配置参数
    input_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhaoziyu-240108120122/xx_help/LLaMA-Factory/data/open-thoughts/OpenThoughts3-1.2M/data"  # 替换为您的输入目录
    base_output_dir = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/zhaoziyu-240108120122/xx_help/LLaMA-Factory/data/open-thoughts/OpenThoughts3-1.2M/data"  # 替换为您的输出目录
    max_samples_per_file = 500  # 每个文件最大处理样本数，None表示处理全部
    
    # 执行批量转换
    batch_convert(
        input_dir=input_dir,
        base_output_dir=base_output_dir,
        format_type="sharegpt",  # 生成instruct和sharegpt两种格式
        max_workers=60,  # 建议设为1避免内存问题
        max_samples=max_samples_per_file
    )