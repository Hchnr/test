import argparse
import os
import time
import tqdm

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l/model.safetensors"
OUT_PATH = "/share/project/hcr/models/wenxinyiyan/ernie45T-4l-Base-PT/model.safetensors"

t_start = time.time()
state_dict = {}
with safe_open(MODEL_PATH, framework="np", device="cpu") as f:
    for key in f.keys():
        weight = f.get_tensor(key)

        # Key: replace "ernie" with "model"
        key = key.replace("ernie", "model")

        # DType:  - UINT16 -> BF16
        if weight.dtype == 'uint16':
            weight = torch.from_numpy(weight).view(torch.bfloat16)
        else:
            weight = torch.from_numpy(weight)

        # Transpose: - weight for proj layers
        if key.endswith('_proj.weight') or key.endswith(".mlp.gate.weight") or key.endswith('lm_head.weight'):
            weight = weight.transpose_(0, 1).contiguous()

        # Split
        print(f"{key:50}: {weight.shape}")
        if ".mlp.experts." in key and ".up_gate_proj." in key:
            assert weight.shape[0] == 3584*2

            gate_key = key.replace(".up_gate_proj.", ".gate_proj.")
            gate_tensor = weight[:3584].contiguous()
            state_dict[gate_key] = gate_tensor

            up_key = key.replace(".up_gate_proj.", ".up_proj.")
            up_tensor = weight[3584:].contiguous()
            state_dict[up_key] = up_tensor
            print(f"{gate_key:50}: {gate_tensor.shape} (gate)")
            print(f"{up_key:50}: {up_tensor.shape} (up)")
        elif ".mlp." in key and ".up_gate_proj." in key:
            assert weight.shape[0] == 28672*2

            gate_key = key.replace(".up_gate_proj.", ".gate_proj.")
            gate_tensor = weight[:28672].contiguous()
            state_dict[gate_key] = gate_tensor

            up_key = key.replace(".up_gate_proj.", ".up_proj.")
            up_tensor = weight[28672:].contiguous()
            state_dict[up_key] = up_tensor
            print(f"{gate_key:50}: {gate_tensor.shape} (gate)")
            print(f"{up_key:50}: {up_tensor.shape} (up)")
        elif ".qkv_proj" in key:
            assert weight.shape[0] == 10240

            q_key = key.replace(".qkv_proj.", ".q_proj.")
            q_tensor = weight[:8192].contiguous()
            state_dict[q_key] = q_tensor

            k_key = key.replace(".qkv_proj.", ".k_proj.")
            k_tensor = weight[8192:8192+1024].contiguous()
            state_dict[k_key] = k_tensor

            v_key = key.replace(".qkv_proj.", ".v_proj.")
            v_tensor = weight[-1024:].contiguous()
            state_dict[v_key] = v_tensor

            print(f"{q_key:50}: {q_tensor.shape} (q)")
            print(f"{k_key:50}: {k_tensor.shape} (k)")
            print(f"{v_key:50}: {v_tensor.shape} (v)")
        else:        
            state_dict[key] = weight

print("="*120)
print(f"{MODEL_PATH} trans success at {OUT_PATH}")
print("Trans Cost: ", time.time() - t_start)
t_start = time.time()
save_file(state_dict, OUT_PATH)
print("Write Cost: ", time.time() - t_start)
