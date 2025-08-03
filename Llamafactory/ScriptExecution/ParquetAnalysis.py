import pyarrow.parquet as pq
import json
from collections import defaultdict
from tqdm import tqdm
import os
import glob
from concurrent.futures import ThreadPoolExecutor


def convert_parquet_analysis_mistakes(parquet_path, output_path):
    """
    分析Parquet文件中评估失败的样本信息
    """
    # 读取Parquet文件
    table = pq.read_table(
        parquet_path,
        memory_map=True,
        use_threads=True,
        buffer_size=1024 * 1024,   # 1MB
    )
    df = table.to_pandas()

    output_data = []

    error_count = 0  # 初始化错误计数器
    total_count = len(df)  # 获取总行数
    total_chars = 0  # 统计总字符数
    current_index = 0  # 添加当前处理行的序号计数器

    for id, row in tqdm(df.iterrows(), total=len(df), desc="Processing Rows"):
        error_data = {}
        current_index += 1  # 更新当前处理的行号
        
        # 处理metrics中的评估数据
        if 'metrics' in row:
            metrics_dict = eval(row['metrics'])  # 将字符串形式的字典转换为实际的字典
            if metrics_dict.get('extractive_match', 1) == 0:  # 提取评分为0的样本
                error_count += 1  # 错误计数加1
                error_data['row_id'] = int(id)
                error_data['sequence_number'] = current_index  # 添加序列号
                error_data['predictions'] = row['predictions']
                # 统计predictions的字符数
                pred_chars = len(str(row['predictions']))
                error_data['prediction_chars'] = pred_chars
                total_chars += pred_chars
    
                # 处理specifics中的extracted_predictions
                if 'specifics' in row and row['specifics']:
                    specifics_dict = eval(row['specifics'])  # 将字符串形式的字典转换为实际的字典
                    error_data['extracted_predictions'] = specifics_dict.get('extracted_predictions', [])
    
                # 只保存有错误数据的样本
                if error_data:
                    output_data.append(error_data)
        
    # 添加统计信息
    statistics = {
        "total_rows": total_count,
        "error_rows": error_count,
        "error_rate": f"{(error_count/total_count)*100:.2f}%",
        "last_processed_row": current_index, # 添加最后处理的行号
        "total_prediction_chars": total_chars,
        "avg_chars_per_error": f"{total_chars/error_count:.2f}" if error_count > 0 else "0"
    }
    output_data.append(statistics)
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)


def batch_convert(input_dir, output_dir, max_workers=4):
    """
    批量转换Parquet文件进行错误分析

    参数：
    input_dir: 包含Parquet文件的输入目录
    output_dir: 输出JSON文件的目录
    max_workers: 并行工作线程数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有Parquet文件（匹配您的命名规范）
    parquet_files = sorted(glob.glob(os.path.join(input_dir, "details_lighteval|*.parquet")))
    print(input_dir)
    # 从第 start_index 个文件开始处理
    start_index = 0
    # 创建处理进度条
    pbar = tqdm(total=len(parquet_files), desc="Processing Files")

    # 错误日志记录器
    error_log = []

    def process_file(file_path):
        try:
            # 生成输出路径
            base_name = os.path.basename(file_path).replace(".parquet", ".json")
            output_path = os.path.join(output_dir, base_name)
            # print("convertion fail")
            # 调用现有转换函数
            convert_parquet_analysis_mistakes(file_path, output_path)

            pbar.update(1)
            return True
        except Exception as e:
            error_log.append(f"Error processing {file_path}: {str(e)}")
            pbar.update(1)
            return False

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, fp) for fp in parquet_files[start_index:]]# 从第 start_index 个文件开始处理

        # 等待所有任务完成
        for future in futures:
            future.result()

    pbar.close()

    # 保存错误日志
    if error_log:
        error_path = os.path.join(output_dir, "conversion_errors.log")
        with open(error_path, "w") as f:
            f.write("\n".join(error_log))
        print(f"完成转换，遇到 {len(error_log)} 个错误，详见 {error_path}")
    else:
        print("所有文件转换成功！")

#批量处理
if __name__ == "__main__":
    # 使用示例（根据硬件配置调整线程数）
    batch_convert(
        input_dir="/home/zdd/xx_help/LLaMA-Factory/eval_result/results/amc23/details/._Model_OriginalModel_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/2025-04-21T03-52-03.198893",  # Parquet文件所在目录
        output_dir="/home/zdd/xx_help/LLaMA-Factory/eval_result/results/amc23/details/._Model_OriginalModel_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/2025-04-21T03-52-03.198893",  # JSON输出目录
        max_workers=8  # 推荐设置为CPU核心数×2
    )
#单个文件处理
# if __name__ == "__main__":
#     # 使用示例
#     convert_parquet_analysis_mistakes(
#         parquet_path="/home/zdd/xx_help/LLaMA-Factory/eval_result/results/aime24/details/._Model_OriginalModel_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/2025-04-18T03-37-55.098786/details_lighteval|aime24|4_2025-04-18T03-37-55.098786.parquet",
#         output_path="/home/zdd/xx_help/LLaMA-Factory/eval_result/results/aime24/details/._Model_OriginalModel_deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/2025-04-18T03-37-55.098786/details_lighteval|aime24|4_2025-04-18T03-37-55.098786.json"
#     )