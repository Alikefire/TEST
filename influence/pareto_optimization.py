import numpy as np
import pandas as pd
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter

# 读取影响因子矩阵
influence_file = '/home/xiexin/xx_help/IDEAL-678C520/influence_outputs/Qwen2.5-0.5B-Instruct-mix_200/influence.csv'
df = pd.read_csv(influence_file, index_col=0)

# 提取影响因子矩阵，跳过第一行（val_avg_grad.pt的数据）
M_IF = df.iloc[1:].values  # 获取数值部分
N_subsets = M_IF.shape[0]  # 数据子集的数量
N_objectives = M_IF.shape[1]  # 目标评估集的数量

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

        # 约束：权重之和约等于1 (pymoo的约束形式 g(x) <= 0)
        # 这里我们用一个简单的约束：权重和偏离1的程度
        out["G"] = [np.abs(np.sum(w) - 1.0)] # 我们希望这个值尽可能小，理想情况是0

problem = InfluenceProblem(M_IF)

# 配置算法 (NSGA-II)
algorithm = NSGA2(
    pop_size=100, # 种群大小
    eliminate_duplicates=True
)

# 执行优化
res = minimize(problem,
               algorithm,
               ('n_gen', 200), # 迭代代数
               seed=1,
               verbose=True)

# 输出结果
print("帕累托最优解 (权重 w):")
print(res.X)
print("对应的目标函数值 (负的预测增益):")
print(res.F)

# 可视化帕累托前沿 (如果目标是2或3维)
if N_objectives == 2 or N_objectives == 3:
    plot = Scatter(title=f"Pareto Front ({N_objectives} Objectives)")
    plot.add(res.F, s=30, facecolors='none', edgecolors='r')
    plot.save('/home/xiexin/xx_help/IDEAL-678C520/ParetoFront.png', dpi=300, bbox_inches='tight')
    plot.show()
elif N_objectives > 3:
    print("目标维度大于3，标准散点图可能不适用。")


# 如何从帕累托前沿选择一个解
# 1. 选择某个目标表现最好的解 (可能牺牲其他目标)
# 2. 选择一个在所有目标上表现相对均衡的解
if res.F is not None and len(res.F) > 0:
    best_idx_obj1 = np.argmin(res.F[:, 0])
    selected_weights_obj1 = res.X[best_idx_obj1]
    selected_gains_obj1 = -res.F[best_idx_obj1]
    print(f"\n选择的权重 (最大化目标1): {selected_weights_obj1}")
    print(f"对应的预测增益: {selected_gains_obj1}")

    # 示例：选择一个离理想点最近的解
    # 假设理想点是所有目标都达到其帕累托前沿上的最大值（这里是最小负增益，即最大正增益）
    # 注意：pymoo 最小化目标，所以 res.F 是负增益
    ideal_point = np.min(res.F, axis=0) # 每个目标在帕累托前沿上的最优值 (最小负增益)
    # 计算每个解到理想点的距离 (欧氏距离)
    distances = np.linalg.norm(res.F - ideal_point, axis=1)
    balanced_idx = np.argmin(distances)
    selected_weights_balanced = res.X[balanced_idx]
    selected_gains_balanced = -res.F[balanced_idx]
    print(f"\n选择的权重 (均衡解 - 离理想点最近): {selected_weights_balanced}")
    print(f"对应的预测增益: {selected_gains_balanced}")
    
    # 创建结果DataFrame
    results_df = pd.DataFrame()
    results_df['Dataset'] = df.index[1:]  # 使用原始数据集的名称，跳过第一行
    results_df['Weight (Best for Obj1)'] = selected_weights_obj1
    results_df['Weight (Balanced)'] = selected_weights_balanced
    
    # 保存结果
    output_file = '/home/xiexin/xx_help/IDEAL-678C520/influence_outputs/Qwen2.5-0.5B-Instruct-mix_200/pareto_weights.csv'
    results_df.to_csv(output_file)
    print(f"\n权重已保存到: {output_file}")
else:
    print("\n未能找到帕累托最优解。")