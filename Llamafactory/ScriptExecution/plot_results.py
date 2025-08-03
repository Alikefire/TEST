import matplotlib.pyplot as plt


def parse_log_file(log_file_path):
    results = {}
    with open(log_file_path, 'r') as file:
        for line in file:
            parts = line.split(':')
            if len(parts) == 2:
                category = parts[0].strip()
                score = float(parts[1].strip())
                results[category] = score
    return results


def plot_results(results):
    categories = list(results.keys())
    scores = list(results.values())
    title='MMLU Task Evaluation Results'
    plt.figure(figsize=(10, 6))
    plt.bar(categories, scores, color='skyblue')
    plt.title(title)
    plt.xlabel('Categories')
    plt.ylabel('Scores')
    plt.ylim(0, 35)  # Adjust the y-axis limit as needed
    # 保存图表
    plt.savefig('/home/zdd/xx_help/LLaMA-Factory/EvaluationPicture/dpo_mmlu.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    log_file_path = '/home/zdd/xx_help/LLaMA-Factory/Model/saves/llama3.2-1B/ppo/lora/eval/ceval/results.log'  # Update with the actual path to your log file
    results = parse_log_file(log_file_path)
    plot_results(results)
