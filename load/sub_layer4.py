from safetensors import safe_open
from safetensors.numpy import save_file
from collections import OrderedDict


MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo/"
FILE_PATHS = [f"model-{i:05d}-of-00121.safetensors" for i in range(1,122)]
# MODEL_PATH = "/share/project/hcr/models/Qwen/Qwen3-0.6B/model.safetensors"
OUTPATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l/model.safetensors"

selected_tensors = OrderedDict()

for file in FILE_PATHS:
    # with safe_open(MODEL_PATH + file, framework="pt") as f:
    with safe_open(MODEL_PATH + file, framework="np") as f:
        for key in f.keys():
            # metadata = f.get_tensor_meta(key)
            # print(f"Shape: {metadata['shape']}")
            # print(f"Data Type: {metadata['dtype']}")
            '''
            print("-" * 50)
            print(f"Tensor Name: {key}")
            print(f"Type       : {type(tensor)}")
            print(f"Shape      : {tensor.shape}")
            print(f"DType      : {tensor.dtype}")
            '''
            if "ernie.layer" in key and "ernie.layers.3." not in key and "ernie.layers.2."not in key and "ernie.layers.1." not in key and "ernie.layers.0." not in key:
                continue
            tensor = f.get_tensor(key)
            selected_tensors[key] = tensor
            print(f"{key:50}: {tensor.shape}")

save_file(selected_tensors, OUTPATH)
