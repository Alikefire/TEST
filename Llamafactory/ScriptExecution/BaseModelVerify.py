from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/zdd/xx_help/LLaMA-Factory/Model/MergeModel/llama3_lora_dpo"
device = "cuda"

# 加载 Base 模型
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 构造补全式输入（Base模型不识别对话格式）
prompt = "The answer of 3+2 is?"  # 替换为你的测试问题

# Base模型无需聊天模板，直接编码文本
inputs = tokenizer(
    prompt, 
    return_tensors="pt", 
    add_special_tokens=True  # 保留模型原始分词逻辑
).to(device)

# 生成参数调整（控制输出相关性）
generated_ids = model.generate(
    **inputs,
    max_new_tokens=100,          # 限制生成长度（避免无限补全）
    temperature=0.7,             # 降低随机性（0~1，越大越随机）
    do_sample=True,              # 启用采样（避免确定性输出）
    eos_token_id=tokenizer.eos_token_id,  # 设置结束符
    pad_token_id=tokenizer.eos_token_id   # 避免警告
)

# 解码时跳过输入部分
output_ids = generated_ids[0][len(inputs.input_ids[0]):]
response = tokenizer.decode(output_ids, skip_special_tokens=True)
print(response)