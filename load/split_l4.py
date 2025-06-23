import torch

from safetensors import safe_open
from safetensors.torch import save_file
from collections import OrderedDict


INPATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch/model.safetensors"
OUTPATH = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split/model.safetensors"

selected_tensors = OrderedDict()

with safe_open(INPATH, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)

        print(f"{key:50}: {tensor.shape}")

        if ".mlp.gate.weight" in key:
            tensor = tensor.transpose_(0, 1).contiguous()
        if "lm_head.weight" in key:
            tensor = tensor.transpose_(0, 1).contiguous()

        if ".moe_statics.e_score_correction_bias" in key:
            key = key.replace(".moe_statics.e_score_correction_bias", ".gate.e_score_correction_bias")
            assert tensor.shape[0] == 1
            tensor = tensor[0]

        if ".mlp.experts." in key and ".up_gate_proj." in key:
            assert tensor.shape[0] == 3584*2

            gate_key = key.replace(".up_gate_proj.", ".gate_proj.")
            gate_tensor = tensor[:3584].contiguous()
            selected_tensors[gate_key] = gate_tensor

            up_key = key.replace(".up_gate_proj.", ".up_proj.")
            up_tensor = tensor[3584:].contiguous()
            selected_tensors[up_key] = up_tensor
            print(f"{gate_key:50}: {gate_tensor.shape} (gate)")
            print(f"{up_key:50}: {up_tensor.shape} (up)")
        else:
            selected_tensors[key] = tensor
        '''
        elif ".mlp.up_gate_proj" in key:
            assert tensor.shape[0] == 57344

            gate_key = key.replace(".up_gate_proj.", ".gate_proj.")
            gate_tensor = tensor[:57344//2].contiguous()
            selected_tensors[gate_key] = gate_tensor

            up_key = key.replace(".up_gate_proj.", ".up_proj.")
            up_tensor = tensor[57344//2:].contiguous()
            selected_tensors[up_key] = up_tensor
            print(f"{gate_key:50}: {gate_tensor.shape} (gate)")
            print(f"{up_key:50}: {up_tensor.shape} (up)")
        '''
        

# selected_tensors["ernie.layers.3.mlp.shared_experts.up_gate_proj.weight"] = torch.zeros(0, 8192)
# selected_tensors["ernie.layers.3.mlp.shared_experts.down_proj.weight"] = torch.zeros(8192, 0)

save_file(selected_tensors, OUTPATH)
