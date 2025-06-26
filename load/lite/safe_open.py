from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Lite/"
FILE_PATHS = [f"model-{i:05d}-of-00009.safetensors" for i in range(1,10)]

for file in FILE_PATHS:
    with safe_open(MODEL_PATH + file, framework="pt") as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            print(f"{key:50}: {tensor.shape} {tensor.dtype}")
