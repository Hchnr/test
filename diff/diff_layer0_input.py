import torch
from safetensors import safe_open


ROOT_PATH = "/share/project/hcr/test/wenxinyiyan/diff/data"
INPUT_FILE = "layer0_input.pt"
NORM_FILE = "layer0_input_norm.pt"

vllm_input_path = f"{ROOT_PATH}/vllm/{INPUT_FILE}"
vllm_norm_path = f"{ROOT_PATH}/vllm/{NORM_FILE}"
fd_input_path = f"{ROOT_PATH}/fastdeploy/{INPUT_FILE}"
fd_norm_path = f"{ROOT_PATH}/fastdeploy/{NORM_FILE}"

tensors = {}
with safe_open(vllm_input_path, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        tensors[f"vllm.{key}"] = tensor

with safe_open(vllm_norm_path, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        tensors[f"vllm.{key}"] = tensor

with safe_open(fd_input_path, framework="np") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        tensor = torch.from_numpy(tensor).view(torch.bfloat16)
        tensors[f"fd.{key}"] = tensor

with safe_open(fd_norm_path, framework="np") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)
        tensor = torch.from_numpy(tensor).view(torch.bfloat16)
        tensors[f"fd.{key}"] = tensor

vi, vn, fi, fn = tensors["vllm.input"], tensors["vllm.input_norm"], tensors["fd.input"], tensors["fd.input_norm"]

print("input,norm shape: ", vi.shape, vn.shape)
print("input_cmp: ", (vi[0] == fi[0]).sum())
print("norm_cmp:  ", (vn[0] == fn[0]).sum())
import pdb; pdb.set_trace()
