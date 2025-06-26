import json
import os

# 文件路径
input_file = "/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct/all_domain_merged.jsonl"
brief_output_file = "/home/xiexin/xx_help/LLaMA-Factory/data/VLyb/s1K-mix/s1_brief_cot.json"
detailed_output_file = "/home/xiexin/xx_help/LLaMA-Factory/data/VLyb/s1K-mix/s1_detailed_cot.json"

# 读取原始JSON文件
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 准备输出数据
brief_output_data = []
detailed_output_data = []

# 处理每个条目
for item in data:
    instruction = item.get('instruction', '')
    
    # 根据instruction开头进行分类
    if instruction.startswith("Answer the problem with a brief thinking process"):
        brief_output_data.append(item)
    elif instruction.startswith("Answer the problem with a detailed thinking process"):
        detailed_output_data.append(item)
    else:
        # 如果不符合上述两种情况，可以选择放入其中一个文件或忽略
        # 这里选择放入detailed_output_data
        detailed_output_data.append(item)

# 保存结果到文件
# with open(brief_output_file, 'w', encoding='utf-8') as f:
#     json.dump(brief_output_data, f, ensure_ascii=False, indent=2)

# with open(detailed_output_file, 'w', encoding='utf-8') as f:
#     json.dump(detailed_output_data, f, ensure_ascii=False, indent=2)

print(f"处理完成！")
print(f"简要思考过程条目数: {len(brief_output_data)}")
print(f"详细思考过程条目数: {len(detailed_output_data)}")