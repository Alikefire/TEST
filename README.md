# TEST - 模型训练与影响力分析工具包

本工具包提供了一套完整的模型训练、聚类分析和影响力分析流程，用于深入理解和优化语言模型的训练过程。

## 📋 环境准备

### 依赖包安装

以下包与 LlamaFactory 环境兼容：

```bash
pip install faiss-gpu
pip install scikit-learn jsonlines
pip install pymoo
pip install psutil
pip install seaborn
```
### 安装说明
将 TEST 目录复制到 LlamaFactory 的根目录下，然后在 LlamaFactory 目录下执行后续操作。

## 🚀 使用流程
### 第一步：模型训练与数据预处理
1.1 数据转换

ParquetConvertInstruct.py用于转换mix-of-thought为alpaca或者sharegpt文件
```
python ./TEST/ParquetConvertInstruct.py
```
1.2 数据预处理
```
python ./TEST/loss_path/sampling_script.py
```
数据分割策略 ：

- 90% 训练集 train（保留原始的目录结构和json(jsonl)文件数目）
- 5% 验证集 validation（不同json文件分别采样、合并后的总体验证集）
- 5% 测试集 evaluation（不同json文件分别采样、合并后的总体测试集）
- validation_cluster是validation的另一版本（不同json文件分别采样,合并前的总体验证集）

1.3 数据合并

MergeJson.py用于合并多个json（jsonl）为一个json(jsonl)文件，对train数据合并后用于模型训练
```
python ./TEST/MergeJson.py
```
1.4 模型训练
1. 训练基础模型
   
   - 使用 SFT 在 mix-of-thought 数据集上训练 Qwen2.5-0.5B 模型
   - 保存约 15 个 checkpoint，
2. 数据格式转换
   
   - 使用 influence/ParquetConvertInstruct.py 将 Parquet 文件转换为 Instruct 格式
   - 重要 ：后续影响力函数计算需要使用 Instruct 格式的 JSONL 数据，建议提前转换
### 第二步：S2L 聚类分析 
2.1 生成损失轨迹

修改/TEST/loss_path/configs/qwen2.5-0.5b_long-short_checkpoint.yml中的模型路径和数据集路径，然后执行：
```
./TEST/loss_path/s2l_distributed_trajectories.sh
```
输出结果 ：

- duplicated_config.json ：记录 losses.pt 文件中的重复样本序号
- 各个 checkpoint 下的 losses.pt 文件 

断点重训：
- specify specific checkpoints，指定参数--checkpoints 1000,2000,3000,4000
- Using a range (start:end:step)，指定参数--checkpoints 1000:5000:1000

2.2 聚类分析与数据分割
```
python ./TEST/loss_path/plot_loss_clusters.py
```
功能 ：

- 生成整体文件的聚类结果
- 生成各子集的聚类结果
- 根据聚类结果将训练集分割为不同的聚类子集，得到的聚类中如果样本数小于10，会增大聚类数目重新聚类，此时需要手动删除那些样本数小于10的聚类以保持

### 第三步：影响力分析 
3.1 运行影响力分析
```
./TEST/influence/run_influence_analysis.sh
```
注意 ： 

- sub-train 参数必须与子集名称保持一致。
- influence分析目前只支持json对象格式为alpaca的的数据

 3.2 Pareto 优化
```
python ./TEST/influence/pareto_optimization.py
```
功能 ：计算 Pareto 前沿下的复杂影响力权重。

 3.3 数据重新加权
```
python ./TEST/influence/reweighting.py
```
功能 ：

- 使用复杂影响力权重对训练集进行重新加权
- 生成 reweighted_data 目录，包含加权后的训练数据
## 📁 输出文件说明
- duplicated_config.json ：重复样本配置文件
- losses.pt ：各 checkpoint 的损失文件
- reweighted_data/ ：重新加权后的训练数据目录
- 聚类结果文件：包含整体和子集的聚类分析结果
## ⚠️ 注意事项
1. 确保所有脚本在 LlamaFactory 根目录下执行
2. 数据格式转换建议在流程开始时完成
3. 子集名称在影响力分析中必须保持一致
4. 建议按顺序执行各个步骤，确保依赖关系正确

# Debug Report
## 06.23
### 代码修改
1. 修正readme文件，将对数据采样与分割的步骤提前到1.2。
2. 改进了patero方法的计算过程，增加了patore方法的多种策略选择 
3. 增加了对影响力矩阵的可视化分析代码
4. 增加ParquetConvertInstruct.py用于转换mix-of-thought为alpaca或者sharegpt文件，MergeJson用于合并多个json（jsonl）为一个json(jsonl)文件
   
### 思路纠正和转变:
1. 纠正之前的讨论过程中忽略的应该何时对数据进行采样和分割的理解，之前的理解是采样和分割需要在聚类完成后。之前考虑的是为了让评估集更可靠，为了提高验证集的质量，所以才选择在loss聚类后才开始对数据集的采样和分割。这样做导致的问题有两个：

- 需要额外的一次SFT过程，浪费了时间。
- 相当于造成了验证集的泄露，因为你是通过训练模型在验证集上表现来选择验证集的。

2. 明晰验证集该怎么选
   
- 之前考虑的是验证集按照聚类提供还是domain提供，经过1点的思路转变后，现在只考虑按照domain选择验证集
- 目前的方法是在域内随机采样数据作为验证集，

4. 分析了phi-4方案的数据混合方法和我们方法的异同点

- 独立优化各领域权重，然后拼接，利用“可加性”保留领域增益 -> 而我们是多领域的共同优化
- 按领域和质量聚类数据源，对于同一聚类内的数据集分配统一权重 -> 和我们的区别在于聚类方法不同，我们是基于loss聚类，不同聚类代表不同的“可学习”程度
