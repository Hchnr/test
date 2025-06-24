import torch

from safetensors import safe_open


FILE_FD = "tmp_tensors_l4.pt"
FILE_VLLM = "new_tmp_tensors_l4.pt"

ts_fd, ts_vllm = {}, {}
with safe_open(FILE_FD, framework="pt") as f:
    for k in f.keys():
        tensor = f.get_tensor(k).cpu()
        ts_fd[k] = tensor
with safe_open(FILE_VLLM, framework="pt") as f:
    for k in f.keys():
        tensor = f.get_tensor(k).cpu()
        ts_vllm[k] = tensor

torch.set_printoptions(precision=8)
for k in ts_fd:
    if k in ts_vllm:
        cmp = ts_fd[k] == ts_vllm[k]
        cmp_res = cmp.sum() == cmp.numel()
        print("=" * 80)
        print(f"{k:40} CHECK_SUCCESS: {cmp_res}", f"eq_num/sum: {cmp.sum()}/{cmp.numel()}")
        print("=" * 60)
        print(ts_fd[k])
        print("-" * 80)
        print(ts_vllm[k])

