import pandas as pd
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

# 读取影响因子矩阵
influence_file = './TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short/influence.csv'
df = pd.read_csv(influence_file, index_col=0)

# 提取影响因子矩阵，跳过第一行（val_avg_grad.pt的数据）并转置
M_IF = df.iloc[1:].values.T  # 转置矩阵：行=训练子集，列=评估集
N_subsets = M_IF.shape[0]  # 训练子集的数量（决策变量数量）
N_objectives = M_IF.shape[1]  # 评估集的数量（目标函数数量）

# 获取评估集名称（子领域名称）
evaluation_domains = df.index[1:].tolist()  # 跳过第一行val_avg_grad.pt
training_subsets = df.columns.tolist()  # 训练子集名称

# 初始化results_df DataFrame
results_df = pd.DataFrame()
results_df['Dataset'] = training_subsets


# 计算目标之间的相关性
corr_matrix = np.corrcoef(M_IF.T)
print(f"\n目标函数之间的相关性矩阵:")
for i, domain1 in enumerate(evaluation_domains):
    for j, domain2 in enumerate(evaluation_domains):
        if i < j:
            print(f"  {domain1} vs {domain2}: {corr_matrix[i,j]:.4f}")

class InfluenceProblem(ElementwiseProblem):
    def __init__(self, m_if_matrix):
        self.m_if_matrix = m_if_matrix
        n_vars = m_if_matrix.shape[0]  # 决策变量的数量等于数据子集的数量
        super().__init__(n_var=n_vars,
                         n_obj=m_if_matrix.shape[1], # 目标数量
                         n_constr=1, # 一个约束：权重之和为1
                         xl=0.0, # 权重下限
                         xu=1.0) # 权重上限

    def _evaluate(self, w, out, *args, **kwargs):
        # w 是一个包含各个数据子集权重的向量
        # 目标是最大化每个领域的预测增益，pymoo默认最小化，所以我们取负值
        predicted_gains = - (w @ self.m_if_matrix) # 矩阵乘法 w * M_IF
        out["F"] = predicted_gains

        # 修改约束处理：使用更宽松的约束
        constraint_violation = abs(np.sum(w) - 1.0)
        out["G"] = [constraint_violation - 0.01]  # 资源约束，我们希望这个值尽可能小，理想情况是0，允许1%的偏差

problem = InfluenceProblem(M_IF)

# 改进的算法配置
algorithm = NSGA2(
    pop_size=200,  # 增加种群大小
    sampling=FloatRandomSampling(),  # 明确指定采样方法
    crossover=SBX(prob=0.9, eta=15),  # 配置交叉算子
    mutation=PM(prob=1.0/N_subsets, eta=20),  # 配置变异算子
    eliminate_duplicates=True
)

# 执行多次优化以增加多样性
print("\n=== 开始多次优化 ===")
all_results = []
for run in range(3):  # 运行3次
    print(f"\n第 {run+1} 次优化...")
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 300),  # 增加迭代次数
                   seed=run,  # 不同的随机种子
                   verbose=False)
    
    if res.F is not None and len(res.F) > 0:
        all_results.append(res)
        print(f"  找到 {len(res.F)} 个帕累托最优解")
        print(f"  目标函数值范围: [{res.F.min():.6f}, {res.F.max():.6f}]")

# 合并所有结果
if all_results:
    # 合并所有运行的结果
    all_X = np.vstack([res.X for res in all_results])
    all_F = np.vstack([res.F for res in all_results])
    
    # 手动进行非支配排序以获得真正的帕累托前沿
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    nds = NonDominatedSorting()
    fronts = nds.do(all_F)
    
    # 取第一前沿（帕累托最优解）
    pareto_indices = fronts[0]
    final_X = all_X[pareto_indices]
    final_F = all_F[pareto_indices]
    
    print(f"\n=== 最终帕累托前沿 ===")
    print(f"帕累托最优解数量: {len(final_X)}")
    print(f"目标函数值范围: [{final_F.min():.6f}, {final_F.max():.6f}]")
    
    # 检查解的多样性
    if len(final_X) > 1:
        diversity = np.std(final_X, axis=0).mean()
        print(f"解的多样性（权重标准差均值）: {diversity:.6f}")
        
        # 如果多样性太低，说明问题可能存在
        if diversity < 0.01:
            print("⚠️  警告：解的多样性很低，可能存在以下问题：")
            print("   1. 目标函数之间高度相关")
            print("   2. 约束过于严格")
            print("   3. 问题本身可能不适合多目标优化")
    else:
        print("⚠️  警告：只找到一个帕累托最优解，这表明：")
        print("   1. 所有目标函数可能指向同一个最优解")
        print("   2. 问题可能退化为单目标优化问题")
    
    # 使用最终结果进行策略选择
    res = type('Result', (), {'X': final_X, 'F': final_F})()
    
    # 1. 新增：为每个评估集添加专门的优化策略
    print(f"\n=== 专门优化策略 ===")
    for i, domain in enumerate(evaluation_domains):
        best_idx = np.argmin(res.F[:, i])  # 选择第i个目标表现最好的解
        selected_weights = res.X[best_idx]
        selected_gains = -res.F[best_idx]
        
        # 添加到results_df
        column_name = f'Weight_优化_{domain}'
        results_df[column_name] = selected_weights
        
        print(f"\n选择的权重 (最大化目标 {domain}): {selected_weights}")
        print(f"对应的预测增益: {selected_gains}")
        print(f"在 {domain} 领域的增益: {selected_gains[i]:.6f}")
    
    # 2. 原有的策略：选择第一个目标表现最好的解（保持向后兼容）
    best_idx_obj1 = np.argmin(res.F[:, 0])
    selected_weights_obj1 = res.X[best_idx_obj1]

    print(f"\n=== 传统策略（向后兼容） ===")
    
    # 3. 均衡解：选择一个离理想点最近的解
    ideal_point = np.min(res.F, axis=0) # 每个目标在帕累托前沿上的最优值 (最小负增益)
    distances = np.linalg.norm(res.F - ideal_point, axis=1)
    balanced_idx = np.argmin(distances)
    selected_weights_balanced = res.X[balanced_idx]
    selected_gains_balanced = -res.F[balanced_idx]
    results_df['Weight_均衡解'] = selected_weights_balanced
    
    print(f"\n选择的权重 (均衡解 - 离理想点最近): {selected_weights_balanced}")
    print(f"对应的预测增益: {selected_gains_balanced}")
    
    # 4. 新增：加权平均策略（用户可以自定义各目标的重要性权重）
    # 示例：假设用户对不同领域有不同的重视程度
    domain_importance = np.ones(N_objectives) / N_objectives  # 默认等权重
    # 用户可以修改这个数组来设置不同领域的重要性，例如：
    # domain_importance = np.array([0.4, 0.3, 0.2, 0.1])  # 假设有4个目标
    
    # 计算加权目标函数值
    weighted_objectives = res.F @ domain_importance  # 加权求和
    weighted_best_idx = np.argmin(weighted_objectives)
    selected_weights_weighted = res.X[weighted_best_idx]
    selected_gains_weighted = -res.F[weighted_best_idx]
    results_df['Weight_加权策略'] = selected_weights_weighted
    
    print(f"\n选择的权重 (加权策略 - 领域重要性权重: {domain_importance}): {selected_weights_weighted}")
    print(f"对应的预测增益: {selected_gains_weighted}")
    
    # 5. 新增：保守策略（最大化最差目标的表现）
    # 这种策略确保在所有领域都有一定的最低保障
    min_gains_per_solution = np.min(-res.F, axis=1)  # 每个解在所有目标上的最小增益
    conservative_idx = np.argmax(min_gains_per_solution)  # 选择最小增益最大的解
    selected_weights_conservative = res.X[conservative_idx]
    selected_gains_conservative = -res.F[conservative_idx]
    results_df['Weight_保守策略'] = selected_weights_conservative
    
    print(f"\n选择的权重 (保守策略 - 最大化最差表现): {selected_weights_conservative}")
    print(f"对应的预测增益: {selected_gains_conservative}")
    print(f"最差领域的增益: {np.min(selected_gains_conservative):.6f}")
    
    # 保存扩展的结果
    output_file = './TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short/pareto_weights_extended.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n扩展的权重结果已保存到: {output_file}")
    
    # 同时保存原格式以保持兼容性
    original_results_df = pd.DataFrame()
    original_results_df['Dataset'] = training_subsets
    original_results_df['Weight (Best for Obj1)'] = selected_weights_obj1
    original_results_df['Weight (Balanced)'] = selected_weights_balanced
    
    original_output_file = './TEST/influence/influence_outputs/Qwen2.5-0.5B-Instruct-long_short/pareto_weights.csv'
    original_results_df.to_csv(original_output_file, index=False)
    print(f"原格式权重已保存到: {original_output_file}")
    
    # 生成策略选择指南
    print(f"\n=== 策略选择指南 ===")
    print(f"1. 专门优化策略: 如果您主要关心某个特定领域的表现，选择对应的'Weight_优化_[领域名]'列")
    for i, domain in enumerate(evaluation_domains):
        print(f"   - Weight_优化_{domain}: 专门优化 {domain} 领域")
    print(f"2. 均衡策略: 如果您希望在所有领域都有较好的表现，选择'Weight_均衡解'列")
    print(f"3. 加权策略: 如果您对不同领域有不同的重视程度，修改domain_importance数组并选择'Weight_加权策略'列")
    print(f"4. 保守策略: 如果您希望避免在任何领域表现过差，选择'Weight_保守策略'列")
    
else:
    print("\n❌ 所有优化运行都失败了，请检查问题设置。")