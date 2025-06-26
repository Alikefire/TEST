#测试模型做单个题目
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
model_path = "./Model/OriginalModel/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
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
prompt = r'''Can sunlight travel to the deepest part of the Black Sea?'''

# CoT
# messages = [
#     {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
#     {"role": "user", "content": prompt}
# ]

# TIR
messages = [
    {"role": "system", "content": "Please integrate natural language reasoning with programs to solve the problem above, and put your final answer within \\boxed{}."},
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
    max_new_tokens=25000,
    do_sample=True,  # 启用采样
    num_beams=1,      # 数值不稳定，使用简单的贪婪搜索
    # sliding_window=None, #Qwen2 模型的注意力机制已经内置了sliding_window
    num_return_sequences=1,
    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
