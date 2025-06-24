import torch

from safetensors import safe_open
from safetensors.torch import save_file


tmp_tensors = {}
with safe_open("tmp_tensors_l4.pd", framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        if tensor.dtype == torch.uint16:
            tensor = tensor.view(torch.bfloat16)

        print(f"{key:50}: {tensor.shape} {tensor.dtype}")
        tmp_tensors[key] = tensor

    save_file(tmp_tensors, "tmp_tensors_l4.pt")
