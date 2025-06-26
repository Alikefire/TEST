# 第一步：创建调整词表后的基础模型
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载原始模型和tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 添加特殊token（与LoRA训练时一致）
tokenizer.add_tokens(["[PAD]", "<mask>"])
model.resize_token_embeddings(len(tokenizer)) 

# 保存调整后的模型
model.save_pretrained("/home/zdd/xx_help/LLaMA-Factory/Model/OriginalModel/meta-llama/Llama2-7b-hf-TokenResize")
tokenizer.save_pretrained("/home/zdd/xx_help/LLaMA-Factory/Model/OriginalModel/meta-llama/Llama2-7b-hf-TokenResize")