import torch
from torch.autograd import Function
import torch.nn as nn
import math
import time
import torch.optim as optim
import numpy as np


class CustomLinearFunction(Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        ctx.save_for_backward(x, weight, bias)
        output = torch.matmul(x, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # 打印反向传播开始的日志
        print("CustomLinearFunction.backward() called")
        
        x, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None
        
        # 计算输入x的梯度，并打印日志
        if ctx.needs_input_grad[0]:
            grad_x = torch.matmul(grad_output, weight)
            # print(f"输入x的梯度形状: {grad_x.shape}")
        
        # 计算权重weight的梯度，并打印日志
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(grad_output.t(), x)
            # print(f"权重weight的梯度形状: {grad_weight.shape}")
            # 可选：打印梯度的部分数值（方便调试）
            # print(f"权重梯度前3个元素: {grad_weight.flatten()[:3]}")
        
        # 计算偏置bias的梯度，并打印日志
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            # print(f"偏置bias的梯度形状: {grad_bias.shape}")
        
        # 打印反向传播结束的日志
        # print("===== Linear层反向传播结束 =====\n")
        return grad_x, grad_weight, grad_bias

# 封装为nn.Module
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        return CustomLinearFunction.apply(x, self.weight, self.bias)


N_LAYER = 2
FROZEN_LAYER = 0
HIDDEN_SIZE = 16384
EPOCHS = 3

np.random.seed(42)
x = np.random.rand(100, HIDDEN_SIZE) * 10  # 100个0-10之间的随机数
y = 2 * x + 3 + np.random.randn(100, HIDDEN_SIZE) * 0.5  # 加入少量噪声

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linears = nn.ModuleList([CustomLinear(in_features=HIDDEN_SIZE, out_features=HIDDEN_SIZE) for _ in range(N_LAYER)])
    
    def forward(self, x):
        for linear in self.linears:
            x = linear(x)
        return x

model = LinearModel()

for n, p in model.named_parameters():
    n_list = [f"linears.{FROZEN_LAYER}.bias", f"linears.{FROZEN_LAYER}.weight"]
    if n in n_list:
        # p.requires_grad = False
        print("freeze: ", n)

criterion = nn.MSELoss()  # 均方误差损失


# opt_params = [p for p in model.parameters() if p.requires_grad]
opt_params = model.parameters()
optimizer = optim.SGD(opt_params, lr=0.01)


print("-" * 40)
print("Params before train:")
for name, param in model.named_parameters():
    if "weight" in name:
        print(f"{name}: {param[0][0].item():.8f}")
print("-" * 40)

for epoch in range(EPOCHS):
    t_s = time.time()
    y_pred = model(x_tensor)
    loss = criterion(y_pred, y_tensor)
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    print(f'Epoch [{epoch}/{EPOCHS}], Loss: {loss.item():.4f}, cost: {(time.time() - t_s)*1000:.1f} ms')

print("-" * 40)
print("Params after train:")
for name, param in model.named_parameters():
    if "weight" in name:
        print(f"{name}: {param[0][0].item():.8f}")
print("-" * 40)
