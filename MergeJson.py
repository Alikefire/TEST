import json
import os
from tqdm import tqdm

def merge_json_files(folder_path, output_file):
    try:
        combined_data = []
        json_files = []
        jsonl_files = []
        
        # 递归遍历所有子目录，收集json和jsonl文件
        for root, dirs, files in os.walk(folder_path):
            for filename in files:
                if filename.endswith('.json'):
                    json_files.append(os.path.join(root, filename))
                elif filename.endswith('.jsonl'):
                    jsonl_files.append(os.path.join(root, filename))
        
        total_files = len(json_files) + len(jsonl_files)
        if total_files == 0:
            print("No JSON or JSONL files found.")
            return
        
        # 创建进度条
        pbar = tqdm(total=total_files, desc="Processing Files")
        
        # 处理JSON文件
        for filepath in json_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend(data)
                    else:
                        combined_data.append(data)
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
            pbar.update(1)
        
        # 处理JSONL文件
        for filepath in jsonl_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            combined_data.append(json.loads(line))
            except Exception as e:
                print(f"Error processing {filepath}: {e}")
            pbar.update(1)
        
        pbar.close()
        
        # 根据输出文件扩展名决定写入格式
        if output_file.endswith('.jsonl'):
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in combined_data:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(f"Successfully merged {total_files} files into {output_file} (JSONL format)")
        else:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(combined_data, f, ensure_ascii=False, indent=2)
            print(f"Successfully merged {total_files} files into {output_file} (JSON format)")
        
        print(f"Total records: {len(combined_data)}")
        
    except Exception as e:
        print(f"Error during merge process: {e}")
        return

if __name__ == '__main__':
    folder_path = '/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/mix_train_data/sharegpt'  # 文件夹路径
    # 示例：输出为JSONL格式
    output_file_jsonl = '/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/mix_train_data/sharegpt/all_domain_merged.jsonl'
    merge_json_files(folder_path, output_file_jsonl)

    # # 示例：输出为JSON格式 (保持原有逻辑)
    # output_file_json = '/home/xiexin/xx_help/LLaMA-Factory/data/open-r1/Mixture-of-Thoughts/mix_train_data/sharegpt_4k/all_domain_merged.json' 
    # merge_json_files(folder_path, output_file_json)
