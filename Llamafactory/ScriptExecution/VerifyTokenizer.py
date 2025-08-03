from transformers import AutoTokenizer

# Qwen2
tokenizer = AutoTokenizer.from_pretrained("/home/zdd/xx_help/LLaMA-Factory/Model/OriginalModel/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
print(tokenizer.tokenize("hello world"))  # 会看到空格前缀

# Llama
llama_tokenizer = AutoTokenizer.from_pretrained("/home/zdd/xx_help/LLaMA-Factory/Model/OriginalModel/meta-llama/Llama2-7b-hf-TokenResize")
print(llama_tokenizer.tokenize("hello world"))  # 会看到下划线前缀