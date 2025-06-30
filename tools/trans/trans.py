import argparse
import os
import time
import tqdm

import torch
from safetensors import safe_open
from safetensors.torch import save_file

MODEL_PATH = "/share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo/"
FILE_PATHS = [f"model-{i:05d}-of-00121.safetensors" for i in range(1,122)]
OUT_PATH = "/share/project/hcr/models/wenxinyiyan/ernie45T-trans/"


def trans(file, model_path, out_path):
    t_start = time.time()
    state_dict = {}
    with safe_open(os.path.join(model_path, file), framework="np", device="cpu") as f:
        for key in f.keys():
            weight = f.get_tensor(key)

            if weight.dtype == 'uint16':
                weight = torch.from_numpy(weight).view(torch.bfloat16)
            else:
                weight = torch.from_numpy(weight)

            if key.endswith('_proj.weight') or key.endswith(".mlp.gate.weight") or key.endswith('lm_head.weight'):
                weight = weight.transpose_(0, 1).contiguous()

            if ".moe_statics.e_score_correction_bias" in key:
                key = key.replace(".moe_statics.e_score_correction_bias", ".gate.e_score_correction_bias")
                assert weight.shape[0] == 1
                weight = weight[0]

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
            else:        
                state_dict[key] = weight
    print("="*120)
    print(f"{file} trans success at {out_path}")
    print("Trans Cost: ", time.time() - t_start)
    t_start = time.time()
    save_file(state_dict, os.path.join(out_path, file))
    print("Write Cost: ", time.time() - t_start)
    


def trans_all():
    for file in tqdm.tqdm(FILE_PATHS, desc="Processing checkpoint..."):
        trans(file, MODEL_PATH, OUT_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
Convert Ernie weights from FastDeploy format to vLLM format.
Example usage:
    python trans.py --file model-00004-of-00121.safetensors --model_path /share/project/hcr/models/wenxinyiyan/paddle_internal/ERNIE-45-Turbo/ --out_path /share/project/hcr/models/wenxinyiyan/ernie45T-trans/
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", type=str, required=True, help="The weights with FastDeploy format")
    parser.add_argument("--model_path", type=str, required=True, help="The weights dir with FastDeploy format")
    parser.add_argument("--out_path", type=str, required=True, help="Output dir")
    args = parser.parse_args()
    trans(args.file, args.model_path, args.out_path)
