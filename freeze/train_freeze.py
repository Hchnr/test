import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

N_LAYER = 2

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
        for linear in self.linears:
            x = linear(x)
        return x

model = LinearModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器


print("-" * 80)
print("Params before train:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")
print("-" * 80)

epochs = 1000
for epoch in range(epochs):
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    
print(f'Epoch [{epochs}], Loss: {loss.item():.4f}')

print("-" * 80)
print("Params after train:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")
print("-" * 80)