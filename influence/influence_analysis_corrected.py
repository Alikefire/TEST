import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import os
warnings.filterwarnings('ignore')

class InfluenceMatrixAnalyzer:
    def __init__(self, csv_path, output_dir='influence_plots'):
        """
        初始化影响力矩阵分析器
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        self.df = None
        self.influence_matrix = None
        self.cluster_matrix = None  # 仅包含聚类数据的矩阵
        self.avg_influence = None   # 平均影响因子行
        self.train_labels = None
        self.val_labels = None
        self.cluster_train_labels = None
        self.cluster_val_labels = None
        self.load_data()
        
    def load_data(self):
        """
        加载和预处理数据，区分平均影响因子和聚类数据
        """
        # 读取CSV文件
        self.df = pd.read_csv(self.csv_path, index_col=0)
        
        # 提取完整影响力矩阵
        self.influence_matrix = self.df.values.astype(float)
        
        # 识别平均影响因子行（第一行）
        avg_row_idx = 0
        for i, idx in enumerate(self.df.index):
            if 'val_avg_grad.pt' in str(idx):
                avg_row_idx = i
                break
        
        # 分离平均影响因子和聚类数据
        self.avg_influence = self.influence_matrix[avg_row_idx, :]
        
        # 创建仅包含聚类数据的矩阵（排除平均行）
        cluster_indices = [i for i in range(len(self.df.index)) if i != avg_row_idx]
        self.cluster_matrix = self.influence_matrix[cluster_indices, :]
        
        # 提取标签信息
        self.val_labels = [self._extract_label(idx) for idx in self.df.index]
        self.train_labels = [self._extract_label(col) for col in self.df.columns]
        
        # 仅聚类的标签
        self.cluster_val_labels = [self.val_labels[i] for i in cluster_indices]
        self.cluster_train_labels = self.train_labels.copy()
        
        print(f"完整影响力矩阵形状: {self.influence_matrix.shape}")
        print(f"聚类影响力矩阵形状: {self.cluster_matrix.shape}")
        print(f"平均影响因子向量长度: {len(self.avg_influence)}")
        print(f"验证集标签数量: {len(self.val_labels)}")
        print(f"训练集标签数量: {len(self.train_labels)}")
        print(f"聚类验证集标签数量: {len(self.cluster_val_labels)}")
        
    def _extract_label(self, path):
        """
        从路径中动态提取数据集和聚类信息。
        """
        filename = os.path.basename(path)

        if 'val_avg_grad.pt' in filename:
            return 'avg_validation'

        # 移除后缀
        if filename.endswith('_grad.pt'):
            base_name = filename[:-8]
        else:
            base_name = os.path.splitext(filename)[0]

        parts = base_name.split('_')

        # 格式: type_dataset_cluster_... or type_dataset_...
        if len(parts) < 2:
            return 'unknown'

        type_prefix = parts[0] # 'train' or 'val'
        
        # 查找 'cluster' 关键字
        try:
            cluster_index = parts.index('cluster')
            dataset_name = '_'.join(parts[1:cluster_index])
            cluster_id = parts[cluster_index + 1]
            return f"{dataset_name}_c{cluster_id}"
        except ValueError:
            # 没有 'cluster'，提取数据集名称
            dataset_name = '_'.join(parts[1:])
            return dataset_name

    def create_heatmap(self, figsize=(15, 12), save_path=None, include_avg=True):
        """
        创建影响力矩阵热力图
        """
        if include_avg:
            matrix_to_plot = self.influence_matrix
            val_labels_to_plot = self.val_labels
            title_suffix = "（包含平均影响因子）"
        else:
            matrix_to_plot = self.cluster_matrix
            val_labels_to_plot = self.cluster_val_labels
            title_suffix = "（仅聚类数据）"
        
        plt.figure(figsize=figsize)
        
        # 创建热力图
        sns.heatmap(matrix_to_plot, 
                   xticklabels=self.train_labels,
                   yticklabels=val_labels_to_plot,
                   annot=True, 
                   fmt='.2e',
                   cmap='RdYlBu_r',
                   center=0,
                   square=True,
                   cbar_kws={'label': '影响力值'})
        
        plt.title(f'训练子集聚类对验证子集聚类的影响力矩阵{title_suffix}', fontsize=16, pad=20)
        plt.xlabel('训练子集聚类', fontsize=12)
        plt.ylabel('验证子集聚类', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            suffix = '_with_avg' if include_avg else '_clusters_only'
            save_path_modified = save_path.replace('.png', f'{suffix}.png')
            plt.savefig(save_path_modified, dpi=300, bbox_inches='tight')
    
    def analyze_average_influence(self, save_path=None):
        """
        分析平均影响因子
        """
        print("\n=== 平均影响因子分析 ===")
        
        # 基本统计
        print(f"\n平均影响因子统计:")
        print(f"  总和: {np.sum(self.avg_influence):.6e}")
        print(f"  平均值: {np.mean(self.avg_influence):.6e}")
        print(f"  标准差: {np.std(self.avg_influence):.6e}")
        print(f"  最大值: {np.max(self.avg_influence):.6e}")
        print(f"  最小值: {np.min(self.avg_influence):.6e}")
        
        # 训练集聚类对整体验证集的影响排序
        print("\n训练集聚类对整体验证集的影响力排序:")
        avg_ranking = sorted(enumerate(self.avg_influence), 
                           key=lambda x: x[1], reverse=True)
        for i, (idx, influence) in enumerate(avg_ranking):
            print(f"  {i+1}. {self.train_labels[idx]}: {influence:.6e}")
        
        # 可视化平均影响因子
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(self.avg_influence)), self.avg_influence)
        plt.xlabel('训练集聚类索引')
        plt.ylabel('平均影响力值')
        plt.title('各训练集聚类对整体验证集的平均影响力')
        plt.xticks(range(len(self.train_labels)), self.train_labels, rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        return avg_ranking
        
    def cross_domain_analysis(self, use_clusters_only=True, save_path=None):
        """
        跨域影响分析（可选择是否包含平均影响因子）
        """
        print(f"\n=== 跨域影响分析 {'（仅聚类数据）' if use_clusters_only else '（包含平均数据）'} ===")
        
        # 选择使用的矩阵和标签
        if use_clusters_only:
            matrix_to_use = self.cluster_matrix
            val_labels_to_use = self.cluster_val_labels
        else:
            matrix_to_use = self.influence_matrix
            val_labels_to_use = self.val_labels
        
        # 创建域映射 (基于新的_extract_label逻辑)
        def get_domain_from_label(label):
            if 'avg_validation' in label:
                return 'avg'
            # 域名是标签中'_c'之前的部分
            return label.split('_c')[0]

        domain_map = {i: get_domain_from_label(label) for i, label in enumerate(val_labels_to_use)}
        train_domain_map = {i: get_domain_from_label(label) for i, label in enumerate(self.train_labels)}
        
        # 计算跨域影响
        unique_domains = sorted(list(set(train_domain_map.values())))
        if not use_clusters_only and 'avg' in set(domain_map.values()):
            val_domains = sorted(list(set(domain_map.values())))
        else:
            val_domains = unique_domains
            
        cross_domain_influence = {}
        
        for val_domain in val_domains:
            for train_domain in unique_domains:
                if val_domain == 'avg':
                    continue
                    
                val_indices = [i for i, d in domain_map.items() if d == val_domain]
                train_indices = [i for i, d in train_domain_map.items() if d == train_domain]
                
                if val_indices and train_indices:
                    influence_values = matrix_to_use[np.ix_(val_indices, train_indices)]
                    avg_influence = np.mean(influence_values)
                    cross_domain_influence[f"{train_domain} -> {val_domain}"] = avg_influence
        
        # 显示结果
        print("\n跨域平均影响力:")
        for key, value in sorted(cross_domain_influence.items(), key=lambda x: x[1], reverse=True):
            print(f"{key}: {value:.6e}")
        
        # 创建跨域影响力矩阵可视化
        self._visualize_cross_domain(cross_domain_influence, unique_domains, save_path)
        
        return cross_domain_influence
    
    def _visualize_cross_domain(self, cross_domain_influence, domains, save_path=None):
        """
        可视化跨域影响力
        """
        # 过滤掉avg域（如果存在）
        plot_domains = [d for d in domains if d != 'avg']
        
        # 创建跨域影响力矩阵
        cross_matrix = np.zeros((len(plot_domains), len(plot_domains)))
        for i, val_domain in enumerate(plot_domains):
            for j, train_domain in enumerate(plot_domains):
                key = f"{train_domain} -> {val_domain}"
                if key in cross_domain_influence:
                    cross_matrix[i, j] = cross_domain_influence[key]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cross_matrix,
                   xticklabels=[f"训练_{d}" for d in plot_domains],
                   yticklabels=[f"验证_{d}" for d in plot_domains],
                   annot=True,
                   fmt='.2e',
                   cmap='RdYlBu_r',
                   square=True,
                   cbar_kws={'label': '平均影响力值'})
        
        plt.title('跨域影响力分析', fontsize=16, pad=20)
        plt.xlabel('训练域', fontsize=12)
        plt.ylabel('验证域', fontsize=12)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def intra_vs_inter_cluster_analysis(self, save_path=None):
        """
        聚类内vs聚类间影响分析（仅使用聚类数据）
        """
        print("\n=== 聚类内vs聚类间影响分析 ===")
        
        # 提取聚类信息
        val_clusters = {}
        train_clusters = {}
        
        for i, label in enumerate(self.cluster_val_labels):
            if '_c' in label:
                dataset, cluster = label.rsplit('_c', 1)
                if dataset not in val_clusters:
                    val_clusters[dataset] = []
                val_clusters[dataset].append((i, int(cluster)))
        
        for i, label in enumerate(self.cluster_train_labels):
            if '_c' in label:
                dataset, cluster = label.rsplit('_c', 1)
                if dataset not in train_clusters:
                    train_clusters[dataset] = []
                train_clusters[dataset].append((i, int(cluster)))
        
        # 计算聚类内和聚类间影响
        intra_influences = []
        inter_influences = []
        
        for val_dataset, val_cluster_list in val_clusters.items():
            if val_dataset in train_clusters:
                train_cluster_list = train_clusters[val_dataset]
                
                for val_idx, val_cluster in val_cluster_list:
                    for train_idx, train_cluster in train_cluster_list:
                        influence_value = self.cluster_matrix[val_idx, train_idx]
                        
                        if val_cluster == train_cluster:
                            intra_influences.append(influence_value)
                        else:
                            inter_influences.append(influence_value)
        
        # 统计分析
        if intra_influences and inter_influences:
            print(f"\n聚类内影响统计:")
            print(f"  数量: {len(intra_influences)}")
            print(f"  平均值: {np.mean(intra_influences):.6e}")
            print(f"  标准差: {np.std(intra_influences):.6e}")
            print(f"  最大值: {np.max(intra_influences):.6e}")
            print(f"  最小值: {np.min(intra_influences):.6e}")
            
            print(f"\n聚类间影响统计:")
            print(f"  数量: {len(inter_influences)}")
            print(f"  平均值: {np.mean(inter_influences):.6e}")
            print(f"  标准差: {np.std(inter_influences):.6e}")
            print(f"  最大值: {np.max(inter_influences):.6e}")
            print(f"  最小值: {np.min(inter_influences):.6e}")
            
            # 比较分析
            ratio = np.mean(intra_influences) / np.mean(inter_influences)
            print(f"\n聚类内/聚类间影响比值: {ratio:.2f}")
            
            # 可视化比较
            self._visualize_intra_vs_inter(intra_influences, inter_influences, save_path)
        
        return intra_influences, inter_influences
    
    def _visualize_intra_vs_inter(self, intra_influences, inter_influences, save_path=None):
        """
        可视化聚类内vs聚类间影响
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 箱线图比较
        data_to_plot = [intra_influences, inter_influences]
        ax1.boxplot(data_to_plot, labels=['聚类内', '聚类间'])
        ax1.set_title('聚类内vs聚类间影响分布')
        ax1.set_ylabel('影响力值')
        ax1.grid(True, alpha=0.3)
        
        # 直方图比较
        ax2.hist(intra_influences, alpha=0.7, label='聚类内', bins=20, density=True)
        ax2.hist(inter_influences, alpha=0.7, label='聚类间', bins=20, density=True)
        ax2.set_title('影响力值分布密度')
        ax2.set_xlabel('影响力值')
        ax2.set_ylabel('密度')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def compare_avg_vs_cluster_influence(self, save_path=None):
        """
        比较平均影响因子与聚类影响因子的差异
        """
        print("\n=== 平均影响因子 vs 聚类影响因子比较 ===")
        
        # 计算每个训练聚类对所有验证聚类的平均影响
        cluster_avg_influence = np.mean(self.cluster_matrix, axis=0)
        
        # 比较平均影响因子和聚类平均影响因子
        print("\n训练聚类影响力比较:")
        print(f"{'训练聚类':<30} {'整体平均影响':<15} {'聚类平均影响':<15} {'差异':<15}")
        print("-" * 75)
        
        for i, train_label in enumerate(self.train_labels):
            avg_inf = self.avg_influence[i]
            cluster_avg_inf = cluster_avg_influence[i]
            diff = avg_inf - cluster_avg_inf
            print(f"{train_label:<30} {avg_inf:<15.6e} {cluster_avg_inf:<15.6e} {diff:<15.6e}")
        
        # 可视化比较
        plt.figure(figsize=(12, 8))
        x = np.arange(len(self.train_labels))
        width = 0.35
        
        plt.bar(x - width/2, self.avg_influence, width, label='整体平均影响', alpha=0.8)
        plt.bar(x + width/2, cluster_avg_influence, width, label='聚类平均影响', alpha=0.8)
        
        plt.xlabel('训练集聚类')
        plt.ylabel('影响力值')
        plt.title('整体平均影响 vs 聚类平均影响比较')
        plt.xticks(x, self.train_labels, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        return cluster_avg_influence
    
    def advanced_analysis(self, use_clusters_only=True):
        """
        高级分析：PCA、聚类分析等
        """
        print(f"\n=== 高级分析 {'（仅聚类数据）' if use_clusters_only else '（包含平均数据）'} ===")
        
        # 选择使用的矩阵和标签
        if use_clusters_only:
            matrix_to_use = self.cluster_matrix
            val_labels_to_use = self.cluster_val_labels
        else:
            matrix_to_use = self.influence_matrix
            val_labels_to_use = self.val_labels
        
        # PCA分析
        print("\n1. 主成分分析 (PCA)")
        pca = PCA(n_components=min(3, matrix_to_use.shape[1]))
        pca_result = pca.fit_transform(matrix_to_use)
        
        print(f"前3个主成分解释的方差比例: {pca.explained_variance_ratio_}")
        print(f"累积解释方差比例: {np.cumsum(pca.explained_variance_ratio_)}")
        
        # 影响力排序分析
        print("\n2. 影响力排序分析")
        total_influence_per_train = np.sum(matrix_to_use, axis=0)
        total_influence_per_val = np.sum(matrix_to_use, axis=1)
        
        print("\n训练集聚类总影响力排序:")
        train_ranking = sorted(enumerate(total_influence_per_train), 
                             key=lambda x: x[1], reverse=True)
        for i, (idx, influence) in enumerate(train_ranking[:5]):
            print(f"  {i+1}. {self.train_labels[idx]}: {influence:.6e}")
        
        print("\n验证集聚类受影响程度排序:")
        val_ranking = sorted(enumerate(total_influence_per_val), 
                           key=lambda x: x[1], reverse=True)
        for i, (idx, influence) in enumerate(val_ranking[:5]):
            print(f"  {i+1}. {val_labels_to_use[idx]}: {influence:.6e}")
        
        # 影响力不对称性分析
        print("\n3. 影响力不对称性分析")
        if matrix_to_use.shape[0] == matrix_to_use.shape[1]:
            asymmetry = matrix_to_use - matrix_to_use.T
            asymmetry_score = np.mean(np.abs(asymmetry))
            print(f"平均不对称性得分: {asymmetry_score:.6e}")
        else:
            print("矩阵不是方阵，无法计算不对称性")
        
        return pca_result, train_ranking, val_ranking
    
    def generate_report(self, save_path=None):
        """
        生成完整分析报告
        """
        print("\n" + "="*50)
        print("         影响力矩阵分析报告")
        print("="*50)
        
        # 基本信息
        print(f"\n数据文件: {self.csv_path}")
        print(f"完整矩阵维度: {self.influence_matrix.shape[0]} × {self.influence_matrix.shape[1]}")
        print(f"聚类矩阵维度: {self.cluster_matrix.shape[0]} × {self.cluster_matrix.shape[1]}")
        print(f"总影响力值（完整）: {np.sum(self.influence_matrix):.6e}")
        print(f"总影响力值（聚类）: {np.sum(self.cluster_matrix):.6e}")
        print(f"平均影响力值（完整）: {np.mean(self.influence_matrix):.6e}")
        print(f"平均影响力值（聚类）: {np.mean(self.cluster_matrix):.6e}")
        
        # 执行各项分析
        print("\n正在执行分析...")
        
        # 1. 平均影响因子分析
        print("\n1. 分析平均影响因子...")
        avg_influence_path = os.path.join(self.output_dir, 'avg_influence.png') 
        avg_ranking = self.analyze_average_influence(save_path=avg_influence_path)
        
        # 2. 热力图（两个版本）
        print("\n2. 生成热力图...")
        heatmap_path_with_avg = os.path.join(self.output_dir, 'heatmap_with_avg.png') 
        self.create_heatmap(save_path=heatmap_path_with_avg, include_avg=True)
        heatmap_path_clusters_only = os.path.join(self.output_dir, 'heatmap_clusters_only.png') 
        self.create_heatmap(save_path=heatmap_path_clusters_only, include_avg=False)
        
        # 3. 平均vs聚类比较
        print("\n3. 比较平均影响因子与聚类影响因子...")
        avg_vs_cluster_path = os.path.join(self.output_dir, 'avg_vs_cluster_comparison.png') 
        cluster_avg_influence = self.compare_avg_vs_cluster_influence(save_path=avg_vs_cluster_path)
        
        # 4. 跨域分析（仅聚类）
        print("\n4. 执行跨域影响分析...")
        cross_domain_path = os.path.join(self.output_dir, 'cross_domain_analysis.png') 
        cross_domain_results = self.cross_domain_analysis(use_clusters_only=True, save_path=cross_domain_path)
        
        # 5. 聚类内外分析
        print("\n5. 执行聚类内vs聚类间分析...")
        intra_inter_path = os.path.join(self.output_dir, 'intra_vs_inter_analysis.png') 
        intra_influences, inter_influences = self.intra_vs_inter_cluster_analysis(save_path=intra_inter_path)
        
        # 6. 高级分析
        print("\n6. 执行高级分析...")
        pca_result, train_ranking, val_ranking = self.advanced_analysis(use_clusters_only=True)
        
        print("\n分析完成！")
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write("影响力矩阵分析报告\n")
                f.write("="*50 + "\n")
                f.write(f"数据文件: {self.csv_path}\n")
                f.write(f"完整矩阵维度: {self.influence_matrix.shape}\n")
                f.write(f"聚类矩阵维度: {self.cluster_matrix.shape}\n")
                f.write(f"总影响力值（完整）: {np.sum(self.influence_matrix):.6e}\n")
                f.write(f"总影响力值（聚类）: {np.sum(self.cluster_matrix):.6e}\n")
                # 添加更多报告内容...
            print(f"\n报告已保存到: {save_path}")

# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = InfluenceMatrixAnalyzer(csv_path='./TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short/influence_cluster.csv',output_dir='./TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short/influence_plot')
    
    # 生成完整分析报告
    analyzer.generate_report()
