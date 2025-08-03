#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JSONL文件聚类分割脚本
基于模型回复长度使用K-means聚类自动分组
支持ShareGPT和Instruction格式
"""

import json
import os
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any


class JsonlClusterSplitter:
    def __init__(self, input_file: str, output_dir: str, n_clusters: int = 5):
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.n_clusters = n_clusters
        
        # 数据存储
        self.records = []
        self.lengths = []
        self.cluster_labels = None
        
        # 统计信息
        self.stats = {
            'total_records': 0,
            'format_counts': {'sharegpt': 0, 'instruction': 0, 'unknown': 0},
            'length_stats': {'min': float('inf'), 'max': 0, 'mean': 0, 'std': 0},
            'cluster_info': {}
        }
        
        # 分组数据存储
        self.grouped_data = defaultdict(list)
    
    def _detect_format_and_extract_response(self, record: Dict[str, Any]) -> Tuple[str, str]:
        """检测格式并提取模型回复内容"""
        # ShareGPT格式
        if 'conversations' in record:
            conversations = record['conversations']
            if isinstance(conversations, list) and conversations:
                # 找到最后一个gpt/assistant回复
                for conv in reversed(conversations):
                    if isinstance(conv, dict) and conv.get('from') in ['gpt', 'assistant']:
                        return 'sharegpt', conv.get('value', '')
            return 'sharegpt', ''
        
        # Instruction格式
        elif 'instruction' in record and 'output' in record:
            return 'instruction', record.get('output', '')
        
        # 其他可能的格式
        elif 'response' in record:
            return 'unknown', record.get('response', '')
        elif 'answer' in record:
            return 'unknown', record.get('answer', '')
        
        return 'unknown', ''
    
    def _count_characters(self, text: str) -> int:
        """计算文本的字符数量"""
        return len(text) if text else 0
    
    def load_and_analyze_data(self):
        """加载数据并分析长度分布"""
        print(f"📖 开始加载和分析文件: {self.input_file}")
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    self.stats['total_records'] += 1
                    
                    # 检测格式并提取回复内容
                    format_type, response_text = self._detect_format_and_extract_response(record)
                    self.stats['format_counts'][format_type] += 1
                    
                    # 计算字符长度
                    char_length = self._count_characters(response_text)
                    
                    # 存储数据
                    record['_char_length'] = char_length
                    record['_format_type'] = format_type
                    record['_response_text'] = response_text
                    
                    self.records.append(record)
                    self.lengths.append(char_length)
                    
                    if line_num % 1000 == 0:
                        print(f"📊 已加载 {line_num} 条记录...")
                
                except json.JSONDecodeError as e:
                    print(f"⚠️ 第{line_num}行JSON解析错误: {e}")
                except Exception as e:
                    print(f"⚠️ 第{line_num}行处理错误: {e}")
        
        # 计算长度统计
        if self.lengths:
            self.lengths = np.array(self.lengths)
            self.stats['length_stats'] = {
                'min': int(np.min(self.lengths)),
                'max': int(np.max(self.lengths)),
                'mean': float(np.mean(self.lengths)),
                'std': float(np.std(self.lengths)),
                'median': float(np.median(self.lengths))
            }
        
        print(f"✅ 数据加载完成，共 {len(self.records)} 条有效记录")
        print(f"📊 长度统计: 最小={self.stats['length_stats']['min']}, 最大={self.stats['length_stats']['max']}, 平均={self.stats['length_stats']['mean']:.1f}")
    
    def perform_clustering(self):
        """执行K-means聚类"""
        if len(self.lengths) == 0:
            print("❌ 没有数据可供聚类")
            return
        
        print(f"🔍 开始执行K-means聚类，分为 {self.n_clusters} 组...")
        
        # 准备数据（将长度转换为二维数组）
        X = self.lengths.reshape(-1, 1)
        
        # 标准化数据（可选，对于一维数据影响不大）
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 执行K-means聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        original_labels = kmeans.fit_predict(X_scaled)
        
        # 获取聚类中心（转换回原始尺度）
        cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_).flatten()
        
        # 计算每个聚类的平均长度并创建排序映射
        cluster_mean_lengths = []
        for i in range(self.n_clusters):
            cluster_mask = original_labels == i
            cluster_lengths = self.lengths[cluster_mask]
            mean_length = float(np.mean(cluster_lengths))
            cluster_mean_lengths.append((i, mean_length))
        
        # 按平均长度排序，获取新的cluster序号映射
        cluster_mean_lengths.sort(key=lambda x: x[1])  # 按平均长度从低到高排序
        
        # 创建原始cluster_id到新cluster_id的映射
        old_to_new_mapping = {}
        for new_id, (old_id, _) in enumerate(cluster_mean_lengths):
            old_to_new_mapping[old_id] = new_id
        
        # 重新分配cluster标签
        self.cluster_labels = np.array([old_to_new_mapping[label] for label in original_labels])
        
        print(f"📊 聚类重排序映射: {old_to_new_mapping}")
        
        # 分析每个聚类（使用新的排序后的ID）
        for new_id in range(self.n_clusters):
            cluster_mask = self.cluster_labels == new_id
            cluster_lengths = self.lengths[cluster_mask]
            
            cluster_info = {
                'count': int(np.sum(cluster_mask)),
                'min_length': int(np.min(cluster_lengths)),
                'max_length': int(np.max(cluster_lengths)),
                'mean_length': float(np.mean(cluster_lengths)),
                'median_length': float(np.median(cluster_lengths)),
                'center': float(cluster_centers[old_to_new_mapping.get(new_id, new_id)])
            }
            
            self.stats['cluster_info'][f'cluster_{new_id}'] = cluster_info
            
            print(f"📊 聚类 {new_id}: {cluster_info['count']} 条记录, 长度范围 [{cluster_info['min_length']}-{cluster_info['max_length']}], 平均 {cluster_info['mean_length']:.1f}")
        
        print("✅ 聚类完成（已按长度排序）")
    
    def group_data_by_clusters(self):
        """根据聚类结果分组数据"""
        print("📁 根据聚类结果分组数据...")
        
        for i, record in enumerate(self.records):
            cluster_id = self.cluster_labels[i]
            cluster_name = f"cluster_{cluster_id}"
            
            # 添加聚类信息到记录
            record['_cluster_id'] = int(cluster_id)
            record['_cluster_name'] = cluster_name
            
            # 分组存储
            self.grouped_data[cluster_name].append(record)
        
        print("✅ 数据分组完成")
    
    def save_grouped_data(self):
        """保存分组数据到不同文件夹"""
        print(f"💾 开始保存分组数据到: {self.output_dir}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for cluster_name, records in self.grouped_data.items():
            if not records:
                continue
            
            # 创建聚类文件夹
            cluster_dir = self.output_dir / cluster_name
            cluster_dir.mkdir(exist_ok=True)
            
            # 保存数据
            output_file = cluster_dir / f"{cluster_name}_data.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    # 移除临时添加的字段（可选）
                    clean_record = {k: v for k, v in record.items() 
                                  if not k.startswith('_')}
                    f.write(json.dumps(clean_record, ensure_ascii=False) + '\n')
            
            cluster_info = self.stats['cluster_info'][cluster_name]
            print(f"📁 {cluster_name}: {len(records)} 条记录 (长度范围: {cluster_info['min_length']}-{cluster_info['max_length']}) -> {output_file}")
    
    def save_statistics(self):
        """保存统计信息"""
        stats_file = self.output_dir / "clustering_statistics.json"
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)
        
        print(f"📊 统计信息已保存到: {stats_file}")
    
    def plot_length_distribution(self):
        """绘制长度分布和聚类结果图"""
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 长度分布直方图
            ax1.hist(self.lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.set_xlabel('Response Length (Characters)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Model Response Length Distribution')
            ax1.grid(True, alpha=0.3)
            
            # 聚类结果散点图
            colors = plt.cm.Set3(np.linspace(0, 1, self.n_clusters))
            for i in range(self.n_clusters):
                cluster_mask = self.cluster_labels == i
                cluster_lengths = self.lengths[cluster_mask]
                y_positions = np.random.normal(i, 0.1, len(cluster_lengths))
                
                ax2.scatter(cluster_lengths, y_positions, 
                           c=[colors[i]], alpha=0.6, s=20, 
                           label=f'Cluster {i} ({np.sum(cluster_mask)} items)')
            
            ax2.set_xlabel('Response Length (Characters)')
            ax2.set_ylabel('Cluster Group')
            ax2.set_title('Clustering Results Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            plot_file = self.output_dir / "length_distribution_and_clustering.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            print(f"📈 分布图已保存到: {plot_file}")
            
            plt.close()
            
        except Exception as e:
            print(f"⚠️ 绘图失败: {e}")
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "="*60)
        print("📊 聚类分析摘要")
        print("="*60)
        print(f"总记录数: {self.stats['total_records']}")
        print(f"聚类数量: {self.n_clusters}")
        
        print(f"\n格式分布:")
        for fmt, count in self.stats['format_counts'].items():
            if count > 0:
                percentage = (count / self.stats['total_records']) * 100
                print(f"  - {fmt}: {count} ({percentage:.1f}%)")
        
        print(f"\n整体长度统计:")
        stats = self.stats['length_stats']
        print(f"  - 最小长度: {stats['min']} 字符")
        print(f"  - 最大长度: {stats['max']} 字符")
        print(f"  - 平均长度: {stats['mean']:.1f} 字符")
        print(f"  - 中位数长度: {stats['median']:.1f} 字符")
        print(f"  - 标准差: {stats['std']:.1f}")
        
        print(f"\n各聚类详情:")
        for cluster_name, info in self.stats['cluster_info'].items():
            print(f"  - {cluster_name}: {info['count']} 条记录")
            print(f"    长度范围: [{info['min_length']}-{info['max_length']}] 字符")
            print(f"    平均长度: {info['mean_length']:.1f} 字符")
            print(f"    中位数长度: {info['median_length']:.1f} 字符")
        
        print("="*60)
    
    def run(self):
        """运行完整流程"""
        self.load_and_analyze_data()
        self.perform_clustering()
        self.group_data_by_clusters()
        self.save_grouped_data()
        self.save_statistics()
        self.plot_length_distribution()
        self.print_summary()


def main():
    parser = argparse.ArgumentParser(
        description="JSONL文件聚类分割工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python JsonlClusterSplitter.py input.jsonl output_dir
  python JsonlClusterSplitter.py input.jsonl output_dir --clusters 3
  python JsonlClusterSplitter.py input.jsonl output_dir --clusters 8
        """
    )
    
    parser.add_argument('--input_file', default= "./data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/science/science_merged.jsonl", help='输入的JSONL文件路径')
    parser.add_argument('--output_dir', default= "./data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct_processed/train/science", help='输出目录路径')
    parser.add_argument('--clusters', '-c', type=int, default=5, 
                       help='聚类数量（默认：5）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_file):
        print(f"❌ 输入文件不存在: {args.input_file}")
        return
    
    # 检查聚类数量
    if args.clusters < 2:
        print(f"❌ 聚类数量必须大于等于2")
        return
    
    # 创建分割器
    splitter = JsonlClusterSplitter(
        input_file=args.input_file,
        output_dir=args.output_dir,
        n_clusters=args.clusters
    )
    
    # 运行处理
    try:
        splitter.run()
        print(f"\n🎉 聚类分割完成！结果保存在: {args.output_dir}")
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()