import matplotlib.pyplot as plt
import numpy as np

def plot_simplified_nested_pie_chart(domain_proportions, cot_proportions_by_domain):
    """
    绘制一个简化的嵌套饼图。
    外环表示各主要领域 (code, math, science) 在数据集中的总体占比。
    内环表示每个主要领域内部的长/短 CoT 比例。

    参数:
    domain_proportions (dict): 描述每个主要领域在总数据集中所占的比例。
                               例如: {'code': 0.4, 'math': 0.35, 'science': 0.25}
                               所有值的总和应为 1。

    cot_proportions_by_domain (dict): 描述每个主要领域内部长短CoT的比例。
                                      键是领域名 (例如 'code'), 值是 {'short_cot':比例, 'long_cot':比例}。
                                      例如: {'code': {'short_cot': 0.7, 'long_cot': 0.3}, 
                                             'math': {'short_cot': 0.4, 'long_cot': 0.6}, ...}
    """

    fig, ax = plt.subplots(figsize=(10, 10))
    size = 0.3 # 内外环的宽度比例
    
    # 外环: 主要领域占比
    outer_labels = list(domain_proportions.keys())
    outer_sizes = list(domain_proportions.values())
    
    # 确保外环比例总和为1
    if not np.isclose(sum(outer_sizes), 1.0):
        print(f"Warning: Sum of domain_proportions is {sum(outer_sizes)}, not 1.0. Normalizing...")
        total = sum(outer_sizes)
        outer_sizes = [s / total for s in outer_sizes]

    outer_colors = plt.cm.Pastel1(np.arange(len(outer_labels)))

    wedges1, texts1, autotexts1 = ax.pie(outer_sizes, labels=outer_labels, autopct='%1.1f%%', startangle=90, 
                                          frame=True, radius=1, colors=outer_colors, 
                                          wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.85)
    plt.setp(autotexts1, size=10, weight="bold")
    plt.setp(texts1, size=12)

    # 内环: 细分每个主要领域的CoT比例
    inner_labels_list = []
    inner_sizes_list = []
    inner_colors_list = []

    # 为 short/long CoT 定义颜色 (可以为每个领域使用基于其外环颜色的深浅变体)
    # 例如: short_cot 用较浅的颜色，long_cot 用较深的颜色

    for i, domain_name in enumerate(outer_labels):
        domain_total_percentage = outer_sizes[i]
        if domain_name in cot_proportions_by_domain:
            cot_details = cot_proportions_by_domain[domain_name]
            
            # 确保每个领域内部CoT比例总和为1
            if not np.isclose(sum(cot_details.values()), 1.0):
                print(f"Warning: CoT proportions for domain '{domain_name}' do not sum to 1. Normalizing...")
                cot_total = sum(cot_details.values())
                cot_details = {k: v / cot_total for k, v in cot_details.items()}

            short_cot_val = cot_details.get('short_cot', 0) * domain_total_percentage
            long_cot_val = cot_details.get('long_cot', 0) * domain_total_percentage
            
            inner_sizes_list.extend([short_cot_val, long_cot_val])
            inner_labels_list.extend([f"{domain_name}\nShort", f"{domain_name}\nLong"]) 
            
            # 基于外环颜色生成内环颜色
            base_color_rgba = outer_colors[i] # RGBA
            # 创建深浅两种颜色
            # Matplotlib颜色通常是 (R, G, B, A) 元组
            # 变浅: 增加亮度, 可以通过将RGB值向1调整 (但不要超过1)
            # 变深: 减少亮度, 可以通过将RGB值向0调整
            # 这里使用一个简单的方法：乘以一个因子
            color_short_cot = np.clip([c * 1.2 for c in base_color_rgba[:3]] + [base_color_rgba[3]], 0, 1) 
            color_long_cot = np.clip([c * 0.7 for c in base_color_rgba[:3]] + [base_color_rgba[3]], 0, 1)
            inner_colors_list.extend([color_short_cot, color_long_cot])
        else:
            # 如果某个外环领域没有内部CoT细分数据，则其在内环中显示为一个整体
            inner_sizes_list.append(domain_total_percentage)
            inner_labels_list.append(domain_name) 
            inner_colors_list.append(outer_colors[i]) # 使用与外环相同的颜色
    
    wedges2, texts2, autotexts2 = ax.pie(inner_sizes_list, autopct='%1.1f%%', startangle=90, 
                                          radius=1-size, colors=inner_colors_list,
                                          wedgeprops=dict(width=size, edgecolor='w'), pctdistance=0.75, labels=None) # 内环不直接显示文字标签
    
    plt.setp(autotexts2, size=8, color='black') #确保百分比数字可见
    # 隐藏内环的文字标签，因为它们可能与外环重叠或难以阅读
    # for t in texts2:
    # t.set_visible(False)

    ax.set(aspect="equal")
    plt.title("Dataset Composition: Domain Proportions and Internal CoT Ratios", fontsize=14, y=1.07)
    
    # 创建一个更清晰的图例来解释内环
    legend_handles = []
    # 为每个领域创建图例条目
    processed_domains_for_legend = set()
    for i, domain_name in enumerate(outer_labels):
        if domain_name in cot_proportions_by_domain and domain_name not in processed_domains_for_legend:
            base_color_rgba = outer_colors[i]
            color_short_cot = np.clip([c * 1.2 for c in base_color_rgba[:3]] + [base_color_rgba[3]], 0, 1) 
            color_long_cot = np.clip([c * 0.7 for c in base_color_rgba[:3]] + [base_color_rgba[3]], 0, 1)
            legend_handles.append(plt.Rectangle((0,0),1,1, color=color_short_cot, label=f'{domain_name} - Short CoT'))
            legend_handles.append(plt.Rectangle((0,0),1,1, color=color_long_cot, label=f'{domain_name} - Long CoT'))
            processed_domains_for_legend.add(domain_name)

    if legend_handles:
        ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10, title="Internal CoT Breakdown")

    plt.tight_layout(rect=[0, 0, 0.85, 1]) #调整布局以容纳图例
    plt.savefig('simplified_nested_domain_cot.png')
    plt.show()

if __name__ == '__main__':
    # 示例数据
    # 1. 主要领域在数据集中的总体占比
    domain_data = {
        'code': 0.45, 
        'math': 0.30, 
        'science': 0.25 
    }

    # 2. 每个主要领域内部的长短CoT比例
    cot_data = {
        'code': {'short_cot': 0.7, 'long_cot': 0.3},
        'math': {'short_cot': 0.4, 'long_cot': 0.6},
        'science': {'short_cot': 0.5, 'long_cot': 0.5}
    }

    plot_simplified_nested_pie_chart(domain_data, cot_data)