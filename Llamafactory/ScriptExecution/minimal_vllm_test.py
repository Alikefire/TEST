import os
from vllm import LLM, SamplingParams

# 确保环境变量设置正确 (或者在运行脚本前 export)
os.environ['VLLM_USE_TRITON'] = '0'
os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 如果只想用单卡测试

# 模型路径
model_path = "/home/zdd/xx_help/LLaMA-Factory/Model/OriginalModel/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
prompts = ["Hello, my name is"] # 简单的测试 prompt

# 加载模型 (设置 tensor_parallel_size=1 模拟单节点多卡中的单卡情况，或根据需要调整)
# 降低 gpu_memory_utilization
try:
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1, # 或者 4 如果你想测试多卡并行
        gpu_memory_utilization=0.7, # 降低内存使用
        dtype='half',
        enforce_eager=True,
        trust_remote_code=True, # 如果模型需要
        enable_chunked_prefill=False  # 禁用分块预填充，v100上不支持
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=50 # 短生成长度测试
    )

    # 生成文本
    outputs = llm.generate(prompts, sampling_params)

    # 打印结果
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    print("Minimal test completed successfully.")

except Exception as e:
    print(f"An error occurred during the minimal test: {e}")
    import traceback
    traceback.print_exc()