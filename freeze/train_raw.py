import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. 生成简单的合成数据集 (y = 2x + 3 + 噪声)
np.random.seed(42)
x = np.random.rand(100, 1) * 10  # 100个0-10之间的随机数
y = 2 * x + 3 + np.random.randn(100, 1) * 0.5  # 加入少量噪声

# 转换为PyTorch张量
x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# 2. 定义最简单的模型 - 线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入维度1，输出维度1的线性层
        self.linear = nn.Linear(in_features=1, out_features=1)
    
    def forward(self, x):
        return self.linear(x)

# 3. 初始化模型、损失函数和优化器
model = LinearModel()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 4. 训练循环
epochs = 1000
for epoch in range(epochs):
    # 前向传播：计算预测值
    y_pred = model(x_tensor)
    
    # 计算损失
    loss = criterion(y_pred, y_tensor)
    
    # 反向传播和参数更新
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    
    # 每100个epoch打印一次信息
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# 5. 打印训练后的模型参数
print("\n训练后的模型参数:")
for name, param in model.named_parameters():
    print(f"{name}: {param.item():.4f}")
