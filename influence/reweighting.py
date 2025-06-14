import json
import pandas as pd
import random
import os
import re

def extract_dataset_and_cluster_info(grad_file_path):
    """
    从梯度文件路径提取数据集名称和聚类信息
    例如：train_gsm8k-r1_train_cluster_1_train_grad.pt
    分解为三个部分：train / gsm8k-r1_train / cluster_1_train
    返回：(top_level_dir, dataset_name, cluster_filename)
    """
    filename = os.path.basename(grad_file_path)
    
    # 移除文件扩展名
    if filename.endswith('_grad.pt'):
        base_name = filename[:-8]  # 移除 '_grad.pt'
    else:
        raise ValueError(f"文件名格式不正确: {filename}")
    
    # 按下划线分割
    parts = base_name.split('_')
    
    if len(parts) < 3:
        raise ValueError(f"文件名格式不正确，无法分解为3级目录: {filename}")
    
    # 第一部分是顶级目录
    top_level_dir = parts[0]  # train
    
    # 找到 'cluster' 的位置
    cluster_index = -1
    for i, part in enumerate(parts):
        if part == 'cluster':
            cluster_index = i
            break
    
    if cluster_index == -1:
        raise ValueError(f"文件名中未找到 'cluster' 关键字: {filename}")
    
    # 数据集名称是从第二部分到cluster之前的所有部分
    dataset_parts = parts[1:cluster_index]
    dataset_name = '_'.join(dataset_parts)  # gsm8k-r1_train
    
    # 聚类文件名是从cluster开始到最后一个train的部分（包含train）
    cluster_parts = parts[cluster_index:]  # ['cluster', '2', 'train']
    cluster_filename = '_'.join(cluster_parts)  # cluster_2_train
    
    return top_level_dir, dataset_name, cluster_filename

def find_json_file(root_dir, top_level_dir, dataset_name, cluster_filename):
    """
    根据数据集名称和聚类名称找到对应的JSON文件
    路径结构：root_dir/top_level_dir/dataset_name/cluster_filename.json 或 cluster_filename.jsonl
    """
    full_dataset_dir = os.path.join(root_dir, top_level_dir, dataset_name)
    
    # 尝试不同的文件扩展名
    for ext in ['.json', '.jsonl']:
        json_file_path = os.path.join(full_dataset_dir, cluster_filename + ext)
        if os.path.exists(json_file_path):
            return json_file_path
    
    raise FileNotFoundError(f"在 {full_dataset_dir} 中找不到 {cluster_filename} 对应的JSON文件")

if __name__ == '__main__':
    # 配置参数
    csv_file_path = "./TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short/pareto_weights.csv"
    root_data_dir = "output_per_dataset_analysis/processed_splits"  # 根据您的描述设置
    target_dir = "output_per_dataset_analysis/reweighted_data"  # 输出目录
    max_num = 0.15  # 最大比例变化
    control_datasum = False  # 是否控制总数据量
    
    # 读取CSV文件，使用第一行作为表头
    df = pd.read_csv(csv_file_path, header=0)
    # 重命名列以便于使用
    df.columns = ['grad_file_path', 'weight1', 'weight2']
    
    # 使用第一个权重列（您可以根据需要选择使用哪个权重）
    weights = df['weight1'].values
    
    # 归一化权重
    if len(weights) > 0:
        beta = max_num / max(abs(w) for w in weights)
        normalized_weights = weights * beta
    else:
        normalized_weights = weights
    
    # 构建影响因子字典
    influence = {}
    file_mapping = {}  # 存储文件路径映射
    
    for i, (grad_path, weight) in enumerate(zip(df['grad_file_path'], normalized_weights)):
        try:
            top_level_dir, dataset_name, cluster_filename = extract_dataset_and_cluster_info(grad_path)
            json_file_path = find_json_file(root_data_dir, top_level_dir, dataset_name, cluster_filename)
            
            key = f"{top_level_dir}_{dataset_name}_{cluster_filename}"
            influence[key] = weight
            file_mapping[key] = json_file_path
            
            print(f"{key}: 调整率={weight:.6f}, 文件路径={json_file_path}")
        except (ValueError, FileNotFoundError) as e:
            print(f"警告：跳过文件 {grad_path}，原因：{e}")
            continue
    
    if not influence:
        print("错误：没有找到有效的数据文件")
        exit(1)
    
    # 读取训练数据
    training_data = {}
    old_sum = 0
    
    for key, json_file_path in file_mapping.items():
        training_data[key] = []
        
        with open(json_file_path, "r", encoding='utf-8') as f:
            if json_file_path.endswith('.jsonl'):
                for line in f:
                    line = line.strip()
                    if line:
                        training_data[key].append(json.loads(line))
            else:  # .json
                data = json.load(f)
                if isinstance(data, list):
                    training_data[key] = data
                else:
                    training_data[key] = [data]
        
        old_sum += len(training_data[key])
        print(f"读取 {key}: {len(training_data[key])} 条数据")
    
    print(f"原始总数据量: {old_sum}")
    
    # 计算调整后的数据量
    new_sum = 0
    adjusted_counts = {}
    
    for key in training_data.keys():
        original_count = len(training_data[key])
        adjusted_count = int(original_count * (1 + influence[key]))
        adjusted_counts[key] = max(1, adjusted_count)  # 确保至少有1条数据
        new_sum += adjusted_counts[key]
    
    print(f"调整后总数据量: {new_sum}")
    
    # 如果需要控制总数据量
    if control_datasum and new_sum > 0:
        for key in adjusted_counts.keys():
            adjusted_counts[key] = int(adjusted_counts[key] * old_sum / new_sum)
            adjusted_counts[key] = max(1, adjusted_counts[key])  # 确保至少有1条数据
    
    # 调整数据
    for key in training_data.keys():
        target_count = adjusted_counts[key]
        current_count = len(training_data[key])
        
        if current_count > target_count:
            # 随机采样减少数据
            training_data[key] = random.sample(training_data[key], target_count)
        elif current_count < target_count:
            # 随机重复增加数据
            additional_needed = target_count - current_count
            additional_data = random.choices(training_data[key], k=additional_needed)
            training_data[key].extend(additional_data)
    
    # 保存调整后的数据
    os.makedirs(target_dir, exist_ok=True)
    
    for key, data in training_data.items():
        # 从key中解析出原始的三个组件
        # key格式：top_level_dir_dataset_name_cluster_filename
        # 例如：train_s1K-mix_s1_brief_cot_cluster_2_train
        
        # 找到第一个下划线，分离top_level_dir
        first_underscore = key.find('_')
        if first_underscore == -1:
            print(f"警告：无法解析key格式: {key}")
            continue
            
        top_level_dir = key[:first_underscore]  # train
        remaining = key[first_underscore + 1:]  # s1K-mix_s1_brief_cot_cluster_2_train
        
        # 在remaining中找到最后一个"_cluster_"的位置
        cluster_pos = remaining.rfind('_cluster_')
        if cluster_pos == -1:
            print(f"警告：在key中找不到cluster标识: {key}")
            continue
            
        dataset_name = remaining[:cluster_pos]  # s1K-mix_s1_brief_cot
        cluster_filename = remaining[cluster_pos + 1:]  # cluster_2_train
        
        # 创建对应的目录结构
        output_top_dir = os.path.join(target_dir, top_level_dir)
        output_dataset_dir = os.path.join(output_top_dir, dataset_name)
        os.makedirs(output_dataset_dir, exist_ok=True)
        
        # 保持原始文件格式
        original_file = file_mapping[key]
        if original_file.endswith('.jsonl'):
            output_file = os.path.join(output_dataset_dir, f"{cluster_filename}.jsonl")
            with open(output_file, "w", encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            output_file = os.path.join(output_dataset_dir, f"{cluster_filename}.json")
            with open(output_file, "w", encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"保存 {key}: {len(data)} 条数据到 {output_file}")
    
    print(f"\n数据重新加权完成！输出目录: {target_dir}")