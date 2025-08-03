import json
import random
import os
from collections import defaultdict
from pathlib import Path

def load_jsonl(file_path):
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def save_jsonl(data, file_path):
    """保存数据为JSONL文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def split_data_into_chunks(data, n_chunks):
    """将数据等分成N份"""
    chunk_size = len(data) // n_chunks
    remainder = len(data) % n_chunks
    
    chunks = []
    start_idx = 0
    
    for i in range(n_chunks):
        # 前remainder个块多分配一个数据点
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        chunks.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    return chunks

def extract_samples_up_to_chunk(file_path, n_chunks, target_chunk, output_path=None, max_samples=None):
    """截取原文件截止到第target_chunk个块的样本，并可选保存到新文件。考虑最大样本限制"""
    if target_chunk < 1 or target_chunk > n_chunks:
        raise ValueError(f"target_chunk必须在1到{n_chunks}之间")
    
    # 设置随机种子以确保可重现性（与main一致）
    random.seed(42)
    
    data = load_jsonl(file_path)
    
    # 应用样本限制（如果设置）
    if max_samples is not None and len(data) > max_samples:
        data = random.sample(data, max_samples)
    
    total_samples = len(data)
    
    chunk_size = total_samples // n_chunks
    remainder = total_samples % n_chunks
    
    # 计算前target_chunk个块的总样本数
    extracted_size = 0
    for i in range(target_chunk):
        extracted_size += chunk_size + (1 if i < remainder else 0)
    
    extracted = data[:extracted_size]
    
    if output_path:
        save_jsonl(extracted, output_path)
        print(f"已保存截取样本到: {output_path} ({len(extracted)} 条)")
    
    return extracted

def mix_data_proportionally(input_files, output_dir, n_chunks, shuffle_within_chunks=True, max_samples_per_file=None):
    """
    将多个JSONL文件按比例混合成N个块，每个块内的混合比例一致
    
    Args:
        input_files: 输入文件路径列表
        output_dir: 输出目录
        n_chunks: 分块数量
        shuffle_within_chunks: 是否在每个块内打乱数据
        max_samples_per_file: 每个文件参与混合的最大样本数，None表示使用全部样本
    """
    # 1. 加载所有数据
    all_data = {}
    total_samples = 0
    
    print("正在加载数据文件...")
    for file_path in input_files:
        domain_name = Path(file_path).stem  # 使用文件名作为域名
        data = load_jsonl(file_path)
        
        # 限制样本数量
        original_count = len(data)
        if max_samples_per_file is not None and len(data) > max_samples_per_file:
            # 随机采样指定数量的样本
            data = random.sample(data, max_samples_per_file)
            print(f"  {domain_name}: {len(data)} 条数据 (从 {original_count} 条中采样)")
        else:
            print(f"  {domain_name}: {len(data)} 条数据")
        
        all_data[domain_name] = data
        total_samples += len(data)
    
    print(f"\n总数据量: {total_samples} 条")
    print(f"将分成 {n_chunks} 个块")
    
    # 2. 将每个域的数据等分成N份
    domain_chunks = {}
    for domain_name, data in all_data.items():
        chunks = split_data_into_chunks(data, n_chunks)
        domain_chunks[domain_name] = chunks
        print(f"\n{domain_name} 分块情况:")
        for i, chunk in enumerate(chunks):
            print(f"  块 {i+1}: {len(chunk)} 条数据")
    
    # 3. 创建N个混合块
    os.makedirs(output_dir, exist_ok=True)
    
    for chunk_idx in range(n_chunks):
        print(f"\n正在创建 Mix_{chunk_idx+1}...")
        
        # 收集当前块的所有数据
        mixed_chunk = []
        chunk_stats = {}
        
        for domain_name in all_data.keys():
            domain_chunk_data = domain_chunks[domain_name][chunk_idx]
            mixed_chunk.extend(domain_chunk_data)
            chunk_stats[domain_name] = len(domain_chunk_data)
        
        # 打乱当前块内的数据（如果需要）
        if shuffle_within_chunks:
            random.shuffle(mixed_chunk)
        
        # 保存当前块
        output_file = os.path.join(output_dir, f"Mix_{chunk_idx+1}.jsonl")
        save_jsonl(mixed_chunk, output_file)
        
        print(f"  Mix_{chunk_idx+1}: {len(mixed_chunk)} 条数据")
        for domain, count in chunk_stats.items():
            percentage = (count / len(mixed_chunk)) * 100
            print(f"    {domain}: {count} 条 ({percentage:.1f}%)")
        print(f"  已保存到: {output_file}")
    
    # 4. 输出总体统计信息
    print("\n=== 混合完成 ===")
    print(f"生成了 {n_chunks} 个混合文件")
    print("每个文件的数据混合比例完全一致")
    
    # 计算总体比例
    total_by_domain = {domain: len(data) for domain, data in all_data.items()}
    print("\n总体数据比例:")
    for domain, count in total_by_domain.items():
        percentage = (count / total_samples) * 100
        print(f"  {domain}: {count} 条 ({percentage:.1f}%)")

def main():
    # 配置参数
    input_files = [
        "./data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/code/code_merged.jsonl",
        "./data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/math/math_merged.jsonl", 
        "./data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/science/science_merged.jsonl"
    ]
    
    output_dir = "./data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/mixed_data"
    n_chunks = 9  # 分成n个块
    
    # 每个文件的最大样本数限制（None表示使用全部样本）
    max_samples_per_file = 64800
    # max_samples_per_file = None  # 使用全部样本
    
    # 设置随机种子以确保可重现性
    random.seed(42)
    
    # 执行数据混合
    # mix_data_proportionally(
    #     input_files=input_files,
    #     output_dir=output_dir,
    #     n_chunks=n_chunks,
    #     shuffle_within_chunks=True,
    #     max_samples_per_file=max_samples_per_file
    # )

    # 提取math.jsonl截止到第n个块的样本
    # 提取并保存math.jsonl截止到第5个块的样本，考虑最大样本限制
    extracted = extract_samples_up_to_chunk(
        './data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/science/science_merged.jsonl',
        9, 5,
        output_path='./science_merged_36k.jsonl',
        max_samples=64800  # 与mix_data_proportionally一致
    )
    print(f"提取了 {len(extracted)} 条样本")
if __name__ == "__main__":
    main()