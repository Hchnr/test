import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

FROZEN = int(os.getenv("FROZEN", 0))
FROZEN_REQUIRE_GRAD = int(os.getenv("FROZEN_REQUIRE_GRAD", 0))
FROZEN_FILER_OPTIMIZER = int(os.getenv("FROZEN_FILER_OPTIMIZER", 0))
FROZEN_NO_GRAD = int(os.getenv("FROZEN_NO_GRAD", 0))
FROZEN_LAYER = int(os.getenv("FROZEN_LAYER", 0))
N_LAYER = 2

print(f"{FROZEN=},{FROZEN_LAYER=},{FROZEN_REQUIRE_GRAD=},{FROZEN_FILER_OPTIMIZER=},{FROZEN_NO_GRAD=}")

np.random.seed(42)
x = np.random.rand(100, 1) * 10  # 100个0-10之间的随机数
y = 2 * x + 3 + np.random.randn(100, 1) * 0.5  # 加入少量噪声

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features=1, out_features=1) for _ in range(N_LAYER)])
    
    def forward(self, x):
        if FROZEN_NO_GRAD:
            if FROZEN_LAYER == 0:
                with torch.no_grad():
                    x = self.linears[0](x)
                x = self.linears[1](x)
            elif FROZEN_LAYER == 1:
                x = self.linears[0](x)
                with torch.no_grad():
                    x = self.linears[1](x)
        else:
            for linear in self.linears:
                x = linear(x)
        return x

model = LinearModel()
if FROZEN and FROZEN_REQUIRE_GRAD:
    for n, p in model.named_parameters():
        n_list = [f"linears.{FROZEN_LAYER}.bias", f"linears.{FROZEN_LAYER}.weight"]
        if n in n_list:
            p.requires_grad = False
            print("freeze: ", n)

criterion = nn.MSELoss()  # 均方误差损失

if FROZEN_FILER_OPTIMIZER:
    opt_params = [p for p in model.parameters() if p.requires_grad]
else:
    opt_params = model.parameters()
optimizer = optim.SGD(opt_params, lr=0.01)


print("-" * 20)
print("Params before train:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")
print("-" * 20)

epochs = 10000

t_s = time.time()
for epoch in range(epochs):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    
print(f'Epoch [{epochs}], Loss: {loss.item():.4f}')
print(f"Train cost: {time.time() - t_s:.1f} s")

print("-" * 20)
print("Params after train:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")
print("-" * 20)
