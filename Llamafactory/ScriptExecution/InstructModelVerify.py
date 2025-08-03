#测试模型做单个题目
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
model_path = "./Model/OriginalModel/prithivMLmods/Theta-Crucis-0.6B-Turbo1"
#
device = "cuda" # the device to load the model onto
# 设置环境变量
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    # attn_implementation="flash_attention_2"  # 使用 Flash Attention 2
    # attn_implementation="sdpa"   # 使用默认的注意力实现
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
#自然问题：How are you.Introduce yourself.
prompt = r"You will be given a competitive programming problem.\nAnalyze the maximum input constraints and identify the optimal algorithmic approach and data structures needed to process the largest possible test cases within the time and memory limits, then explain why your chosen implementation strategy is the most efficient solution. Please reason step by step about your solution approach, then provide a complete implementation in Python 3 that is thoroughly optimized for both speed and memory usage.\n\nYour solution must read input from standard input (input()), write output to standard output (print()).\nDo not include any debug prints or additional output.\n\nPut your final solution within a single code block:\n```python\n<your code here>\n```\n\n# Problem\n\nIt is never too late to play the fancy \"Binary Cards\" game!\n\nThere is an infinite amount of cards of positive and negative ranks that are used in the game. The absolute value of any card rank is a power of two, i.e. each card has a rank of either 2k or - 2k for some integer k ≥ 0. There is an infinite amount of cards of any valid rank.\n\nAt the beginning of the game player forms his deck that is some multiset (possibly empty) of cards. It is allowed to pick any number of cards of any rank but the small deck is considered to be a skill indicator. Game consists of n rounds. In the i-th round jury tells the player an integer ai. After that the player is obligated to draw such a subset of his deck that the sum of ranks of the chosen cards is equal to ai (it is allowed to not draw any cards, in which case the sum is considered to be equal to zero). If player fails to do so, he loses and the game is over. Otherwise, player takes back all of his cards into his deck and the game proceeds to the next round. Player is considered a winner if he is able to draw the suitable set of cards in each of the rounds.\n\nSomebody told you which numbers ai the jury is going to tell you in each round. Now you want to pick a deck consisting of the minimum number of cards that allows you to win the \"Binary Cards\" game.\n\n## Constraints\nTime limit per test: 1.0 seconds\nMemory limit per test: 512.0 megabytes\n\n## Input Format\nThe first line of input contains an integer n (1 ≤ n ≤ 100 000), the number of rounds in the game.\n\nThe second line of input contains n integers a1, a2, ..., an ( - 100 000 ≤ ai ≤ 100 000), the numbers that jury is going to tell in each round.\n\n## Output Format\nIn the first line print the integer k (0 ≤ k ≤ 100 000), the minimum number of cards you have to pick in your deck in ordered to win the \"Binary Cards\".\n\nIn the second line print k integers b1, b2, ..., bk ( - 220 ≤ bi ≤ 220, |bi| is a power of two), the ranks of the cards in your deck. You may output ranks in any order. If there are several optimum decks, you are allowed to print any of them.\n\nIt is guaranteed that there exists a deck of minimum size satisfying all the requirements above.\n\n## Examples\n```input\n1\n9\n```\n```output\n2\n1 8\n```\n-----\n```input\n5\n-1 3 0 4 7\n```\n```output\n3\n4 -1 4\n```\n-----\n```input\n4\n2 -2 14 18\n```\n```output\n3\n-2 2 16\n```\n\n## Note\nIn the first sample there is the only round in the game, in which you may simply draw both your cards. Note that this sample test is the only one satisfying the first test group constraints.\n\nIn the second sample you may draw the only card - 1 in the first round, cards 4 and - 1 in the second round, nothing in the third round, the only card 4 in the fourth round and the whole deck in the fifth round.\n"

# CoT
# messages = [
#     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt}
# ]

#system:Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}.
# TIR
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=16384,
    do_sample=True,  # 启用采样
    temperature=0.6,
    top_k=20,
    top_p=0.95,
    repetition_penalty= 1.1,
    no_repeat_ngram_size= 10,
    # num_beams=1,      # 数值不稳定，使用简单的贪婪搜索
    # sliding_window=None, #Qwen2 模型的注意力机制已经内置了sliding_window
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
