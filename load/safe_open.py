from safetensors import safe_open

MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo/"
MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split/"
FILE_PATHS = [f"model-{i:05d}-of-00121.safetensors" for i in range(1,122)]
FILE_PATHS = ["model.safetensors"]
# MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"

for file in FILE_PATHS:
    with safe_open(MODEL_PATH + file, framework="pt") as f:
        for key in f.keys():
            # metadata = f.get_tensor_meta(key)
            # print(f"Shape: {metadata['shape']}")
            # print(f"Data Type: {metadata['dtype']}")
            tensor = f.get_tensor(key)
            '''
            print("-" * 50)
            print(f"Tensor Name: {key}")
            print(f"Type       : {type(tensor)}")
            print(f"Shape      : {tensor.shape}")
            print(f"DType      : {tensor.dtype}")
            '''

            print(f"{key:50}: {tensor.shape} {tensor.dtype}")
