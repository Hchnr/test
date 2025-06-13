from fastdeploy import LLM, SamplingParams

MODEL = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l"

prompts = ["The largest ocean is",]

# 采样参数
sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=2)

# 加载模型
llm = LLM(model=MODEL, tensor_parallel_size=1, max_model_len=8192, gpu_memory_utilization=0.2)

# 批量进行推理（llm内部基于资源情况进行请求排队、动态插入处理）
outputs = llm.generate(prompts, sampling_params)

import pdb; pdb.set_trace()

# 输出结果
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs.text
    print(f"Prompt: {prompt}")
    print(f"Generated Text: {generated_text}")

import pdb; pdb.set_trace()
