import pyarrow.parquet as pq
import pandas as pd
# 读取 Parquet 文件
table = pq.read_table('/home/zdd/xx_help/LLaMA-Factory/data/open-r1/OpenR1-Math-220k/data/train-00000-of-00010.parquet')
df=table.to_pandas()
# 查看 Schema
print(table.schema)
print(table.num_rows)
# 查看前几行数据
# print(df.head(1))
# 查看第5行（索引从0开始）
# target_row = df.iloc[4]  # 访问第五行数据
# print(target_row)

# 查看满足条件的行（如id=100）
# filtered_rows = df[df['id'] == 100]
# print(filtered_rows)
# from prettytable import PrettyTable

# 创建表格对象
# pt = PrettyTable()
# pt.field_names = ["Column", "Value"]
#
# # 添加数据
# row_data = table.take([98]).to_pylist()[0]#读取第100行
# for k, v in row_data.items():
#     pt.add_row([k, str(v)[:1000] + "..."])  # 截断长文本
#
# print(pt)