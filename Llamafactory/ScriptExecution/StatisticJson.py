import json
from collections import Counter

def analyze_gpt_response_lengths(file_path):
    """
    分析 JSON 文件中 'gpt' 回复的长度分布，并找出最长的5条回复及其序号。

    Args:
        file_path (str): JSON 文件的路径。

    Returns:
        collections.Counter: 一个计数器，键是长度区间（以10000为单位的起始值），
                             值是该区间内 'gpt' 回复的数量。
                             例如：key 0 表示长度 0-9999，key 1 表示长度 10000-19999。
    """
    interval_counts = Counter()
    max_len = 0 # 用于追踪最大长度，方便确定需要打印多少区间
    gpt_responses_details = [] # 用于存储 (length, item_index, turn_index)

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item_index, item in enumerate(data): # item_index 是顶层JSON对象的序号
            if "conversations" in item and isinstance(item["conversations"], list):
                for turn_index, conversation_turn in enumerate(item["conversations"]): # turn_index 是conversations列表内的序号
                    if isinstance(conversation_turn, dict) and \
                       conversation_turn.get("from") == "gpt" and \
                       "value" in conversation_turn and \
                       isinstance(conversation_turn["value"], str):
                        
                        value_str = conversation_turn["value"]
                        length = len(value_str)
                        
                        gpt_responses_details.append((length, item_index, turn_index))
                        
                        if length > max_len:
                            max_len = length
                        
                        # 计算所属区间
                        interval_key = length // 10000 # 保持原有的区间统计逻辑
                        interval_counts[interval_key] += 1
        
        print(f"成功处理文件: {file_path}")
        print("GPT 回复长度分布统计 (每10000字符为一个区间):")
        
        if not interval_counts:
            print("没有找到来自 'gpt' 的回复，或者回复格式不符合预期。")
            return interval_counts

        # 确定需要打印到哪个区间
        max_interval_key = max_len // 10000
        
        for i in range(max_interval_key + 1):
            lower_bound = i * 10000
            upper_bound = (i + 1) * 10000 - 1
            count = interval_counts.get(i, 0) # 如果某个区间没有数据，则数量为0
            print(f"长度区间 {lower_bound}-{upper_bound}: {count} 条")

        # 排序并打印最长的5条GPT回复的详细信息
        gpt_responses_details.sort(key=lambda x: x[0], reverse=True)
        
        print("\n长度最大的5条GPT回复及其序号:")
        for i in range(min(5, len(gpt_responses_details))):
            length, item_idx, turn_idx = gpt_responses_details[i]
            print(f"  Top {i+1}: 长度 = {length}, "
                  f"位于顶层对象序号 = {item_idx}, "
                  f"该对象内对话序号 = {turn_idx}")
            # 如果需要，可以打印部分内容以供参考
            # original_item = data[item_idx]
            # gpt_value_preview = original_item["conversations"][turn_idx]["value"][:100] # 打印前100个字符
            # print(f"     内容预览: \"{gpt_value_preview}...\"")

    except FileNotFoundError:
        print(f"错误: 文件未找到 {file_path}")
    except json.JSONDecodeError:
        print(f"错误: 解析 JSON 文件失败 {file_path}")
    except Exception as e:
        print(f"发生未知错误: {e}")
        
    return interval_counts

if __name__ == "__main__":
    # 请将下面的文件路径替换为您实际的 train.json 文件路径
    file_to_analyze = "/home/xiexin/xx_help/LLaMA-Factory/data/simplescaling/s1K-1.1/data/train.json"
    
    analysis_results = analyze_gpt_response_lengths(file_to_analyze)
    # analysis_results 变量中会存储统计结果，如果需要后续处理可以使用