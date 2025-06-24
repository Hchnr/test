from safetensors import safe_open


# FILE = "tmp_tensors_l4.pd"
# FILE = "tmp_tensors_l4.pt"
FILE = "new_tmp_tensors_l4.pt"

with safe_open(FILE, framework="pt") as f:
    for key in f.keys():
        tensor = f.get_tensor(key)

        print(f"{key:50}: {tensor.shape} {tensor.dtype}")
