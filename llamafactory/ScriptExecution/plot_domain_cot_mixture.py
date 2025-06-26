import matplotlib.pyplot as plt
import numpy as np

def plot_domain_cot_mixture(domain_proportions, inter_domain_matrix=None):
    """
    绘制领域内长/短 CoT 混合比例以及可选的领域间混合比例图。

    参数:
    domain_proportions (dict): 一个字典，键是领域名称 (例如 'code', 'math', 'science')，
                               值是另一个字典，包含 'short_cot' 和 'long_cot' 的比例。
                               例如: 
                               {
                                   'code': {'short_cot': 0.7, 'long_cot': 0.3},
                                   'math': {'short_cot': 0.4, 'long_cot': 0.6},
                                   'science': {'short_cot': 0.5, 'long_cot': 0.5}
                               }
    inter_domain_matrix (dict, optional): 一个字典，表示领域间的混合比例。
                                        键是源领域，值是另一个字典，目标领域为键，比例为值。
                                        例如: 
                                        {
                                            'code': {'math': 0.1, 'science': 0.05},
                                            'math': {'code': 0.08, 'science': 0.12},
                                            'science': {'code': 0.07, 'math': 0.1}
                                        }
                                        如果为 None，则不绘制领域间混合图。
    """
    
    domains = list(domain_proportions.keys())
    short_cot_proportions = [domain_proportions[d]['short_cot'] for d in domains]
    long_cot_proportions = [domain_proportions[d]['long_cot'] for d in domains]

    x = np.arange(len(domains))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, short_cot_proportions, width, label='Short CoT')
    rects2 = ax.bar(x + width/2, long_cot_proportions, width, label='Long CoT')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Proportion')
    ax.set_title('Intra-Domain CoT Mixture Proportions')
    ax.set_xticks(x)
    ax.set_xticklabels(domains)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()
    plt.savefig('intra_domain_cot_mixture.png') # 保存领域内比例图
    plt.show()

    if inter_domain_matrix:
        # 对于领域间混合，热力图可能是一个好的选择
        num_domains = len(domains)
        matrix_data = np.zeros((num_domains, num_domains))
        domain_to_idx = {name: i for i, name in enumerate(domains)}

        for i, source_domain in enumerate(domains):
            if source_domain in inter_domain_matrix:
                for target_domain, proportion in inter_domain_matrix[source_domain].items():
                    if target_domain in domain_to_idx:
                        j = domain_to_idx[target_domain]
                        matrix_data[i, j] = proportion
        
        fig_inter, ax_inter = plt.subplots(figsize=(8, 7))
        cax = ax_inter.matshow(matrix_data, cmap='viridis')
        fig_inter.colorbar(cax)

        ax_inter.set_xticks(np.arange(num_domains))
        ax_inter.set_yticks(np.arange(num_domains))
        ax_inter.set_xticklabels(domains)
        ax_inter.set_yticklabels(domains)
        plt.setp(ax_inter.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")

        for i in range(num_domains):
            for j in range(num_domains):
                if matrix_data[i, j] > 0:
                    ax_inter.text(j, i, f'{matrix_data[i, j]:.2f}', va='center', ha='center', color='white' if matrix_data[i,j] < 0.5 else 'black')

        ax_inter.set_title('Inter-Domain Mixture Proportions (Source to Target)')
        ax_inter.set_xlabel('Target Domain')
        ax_inter.set_ylabel('Source Domain')
        fig_inter.tight_layout()
        plt.savefig('inter_domain_mixture_heatmap.png') # 保存领域间混合热力图
        plt.show()


if __name__ == '__main__':
    # 示例数据
    intra_domain_data = {
        'code': {'short_cot': 0.7, 'long_cot': 0.3},
        'math': {'short_cot': 0.4, 'long_cot': 0.6},
        'science': {'short_cot': 0.5, 'long_cot': 0.5}
    }

    # 示例领域间混合数据 (可选)
    # 表示从源领域到目标领域的混合比例
    inter_domain_data = {
        'code': {'math': 0.1, 'science': 0.05}, # code 数据中，有10%是与math混合，5%与science混合
        'math': {'code': 0.08, 'science': 0.12},
        'science': {'code': 0.07, 'math': 0.1}
    }

    plot_domain_cot_mixture(intra_domain_data, inter_domain_data)

    # 如果只想看领域内比例
    # plot_domain_cot_mixture(intra_domain_data)