import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

N_LAYER = 2
FROZEN_LAYER = 0
HIDDEN_SIZE = 16384
EPOCHS = 20

np.random.seed(42)
x = np.random.rand(100, HIDDEN_SIZE) * 10  # 100个0-10之间的随机数
y = 2 * x + 3 + np.random.randn(100, HIDDEN_SIZE) * 0.5  # 加入少量噪声

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE) for _ in range(N_LAYER)])
    
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

model = LinearModel()

for n, p in model.named_parameters():
    n_list = [f"linears.{FROZEN_LAYER}.bias", f"linears.{FROZEN_LAYER}.weight"]
    if n in n_list:
        p.requires_grad = False
        print(f"freeze:{n} requires_grad:{p.requires_grad} leaf:{p.is_leaf}")

criterion = nn.MSELoss()  # 均方误差损失


origin_params = [p for p in model.parameters()]
opt_params = [p for p in origin_params if p.requires_grad]
# opt_params = model.parameters()
print(f"optimizer after filter: {len(opt_params)}/{len(origin_params)}")
optimizer = optim.SGD(opt_params, lr=0.01)


print("-" * 20)
print("Params before train:")
for name, param in model.named_parameters():
    print(f"{name}: {param[0]}")
print("-" * 20)

for epoch in range(EPOCHS):
    t_s = time.time()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    print(f'Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.4f}, cost: {(time.time() - t_s)*1000:.1f} ms')

print("-" * 20)
print("Params after train:")
for name, param in model.named_parameters():
    print(f"{name}: {param[0]}")
print("-" * 20)
