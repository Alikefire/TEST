def count_jsonl_lines(file_path):
    """
    统计 JSONL 文件中的总行数。

    参数:
        file_path (str): JSONL 文件的路径。

    返回:
        int: 文件中的总行数。
    """
    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for _ in f:
                line_count += 1
    except FileNotFoundError:
        print(f"错误：文件 '{file_path}' 未找到。")
        return -1  # 或者可以抛出异常
    except Exception as e:
        print(f"读取文件时发生错误：{e}")
        return -1  # 或者可以抛出异常
    return line_count

# 示例用法：
file_to_count = "/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/mix_train_data/instruct/all_domain_merged.jsonl"
total_lines = count_jsonl_lines(file_to_count)
if total_lines != -1:
    print(f"文件 '{file_to_count}' 中的总行数为: {total_lines}")
