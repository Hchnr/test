# torchrun --nproc_per_node=2 /share/project/hcr/repos/test_gh/test/freeze/train1_0.py
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

N_LAYER = 2
FROZEN_LAYER = 0
HIDDEN_SIZE = 16384
EPOCHS = 20

def setup_distributed():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

def cleanup_distributed():
    torch.distributed.destroy_process_group()

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE) for _ in range(N_LAYER)])
    
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

def main():
    setup_distributed()

    np.random.seed(42)
    x = np.random.rand(100, HIDDEN_SIZE) * 10
    y = 2 * x + 3 + np.random.randn(100, HIDDEN_SIZE) * 0.5

    x_tensor = torch.tensor(x, dtype=torch.float32).cuda()
    y_tensor = torch.tensor(y, dtype=torch.float32).cuda()

    model = LinearModel().cuda()

    for n, p in model.named_parameters():
        n_list = [f"linears.{FROZEN_LAYER}.bias", f"linears.{FROZEN_LAYER}.weight"]
        if n in n_list:
            # p.requires_grad = False
            if int(os.environ["LOCAL_RANK"]) == 0:
                print("freeze: ", n)

    model = FSDP(
        model,
        use_orig_params=True,
    )

    criterion = nn.MSELoss()
    print("requires_grad: ", [p.requires_grad for p in model.parameters()])
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(opt_params, lr=0.01)

    if int(os.environ["LOCAL_RANK"]) == 0:
        print("-" * 20)
        print("Params before train:")
        for name, param in model.named_parameters():
            if "weight" in name:
                # print(f"{name}: {param[0][0].item():.8f}")
                print(f"{name}: {param.shape} {param}")
        print("-" * 20)

    for epoch in range(EPOCHS):
        t_s = time.time()
        y_pred = model(x_tensor)
        loss = criterion(y_pred, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if int(os.environ["LOCAL_RANK"]) == 0:
            print(f'Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.4f}, cost: {(time.time() - t_s)*1000:.1f} ms')

    if int(os.environ["LOCAL_RANK"]) == 0:
        print("-" * 20)
        print("Params after train:")
        for name, param in model.named_parameters():
            if "weight" in name:
                # print(f"{name}: {param[0][0].item():.8f}")
                print(f"{name}: {param.shape} {param}")
        print("-" * 20)

    cleanup_distributed()

if __name__ == "__main__":
    main()
