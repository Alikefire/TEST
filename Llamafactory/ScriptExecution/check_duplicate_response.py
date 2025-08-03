import json
import sys
from collections import Counter
import re

def detect_format(data):
    if isinstance(data, list):
        first_item = data[0] if data else {}
        if 'response' in first_item:
            return 'response'
        elif 'output_list' in first_item:
            return 'output_list'
    return None

def split_into_sentences(text):
    # 简单句子分割，支持中英文
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|。|？|！)\\s', text)
    return [s.strip() for s in sentences if s.strip()]

def check_duplicates_in_content(content):
    sentences = split_into_sentences(content)
    total_sentences = len(sentences)
    if total_sentences == 0:
        return 0.0, {}
    
    sentence_counts = Counter(sentences)
    duplicates = {s: count for s, count in sentence_counts.items() if count > 1}
    duplicate_count = sum(count - 1 for count in duplicates.values())  # 每个重复的额外出现
    duplicate_ratio = (duplicate_count / total_sentences) * 100 if total_sentences > 0 else 0
    return duplicate_ratio, duplicates

def main(json_file):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return
    
    format_type = detect_format(data)
    if not format_type:
        print("Unsupported JSON format.")
        return
    
    total_items = len(data)
    duplicate_items = 0
    report = []
    
    for idx, item in enumerate(data, 1):
        contents = []
        if format_type == 'response':
            if 'response' in item:
                contents = [item['response']]
        elif format_type == 'output_list':
            if 'output_list' in item:
                contents = item['output_list']
        
        has_duplicate = False
        item_report = f"Item {idx}:"
        for sub_idx, content in enumerate(contents, 1):
            ratio, dups = check_duplicates_in_content(content)
            if dups:
                has_duplicate = True
                item_report += f"\n  Sub-item {sub_idx}: Duplicate ratio: {ratio:.2f}%"
                for sentence, count in dups.items():
                    item_report += f"\n    Repeated sentence (x{count}): {sentence}"
        
        if has_duplicate:
            duplicate_items += 1
            report.append(item_report)
    
    overall_ratio = (duplicate_items / total_items) * 100 if total_items > 0 else 0
    print(f"Overall: {duplicate_items}/{total_items} items have internal duplicates ({overall_ratio:.2f}%)")
    if report:
        print("\nDetailed report:")
        for r in report:
            print(r + "\n")
    else:
        print("No internal duplicates found.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_duplicates_in_response.py <json_file>")
        sys.exit(1)
    main(sys.argv[1])