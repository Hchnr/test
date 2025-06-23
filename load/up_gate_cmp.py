from safetensors import safe_open

MODEL_PATH1 = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l_torch_split/model.safetensors"
MODEL_PATH2 = "/share/project/hcr/models/wenxinyiyan/ernie34T-4l/model.safetensors"

KEY = "ernie.layers.0.mlp.up_gate_proj.weight"

with safe_open(MODEL_PATH1, framework="pt") as f1:
    t1 = f1.get_tensor(KEY)
    
with safe_open(MODEL_PATH1, framework="pt") as f2:
    t2 = f2.get_tensor(KEY)

import pdb; pdb.set_trace()
