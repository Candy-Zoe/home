# PyTorch自动微分学习
# 主要内容：autograd机制、梯度计算、requires_grad参数

import torch

print("=== 自动微分基础 ===")

# 创建带有梯度追踪的张量
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

print(f"x = {x}")
print(f"y = x^2 + 3x + 1 = {y}")

# 计算梯度
y.backward()
print(f"dy/dx 在 x=2 处的值: {x.grad}")

print("\n=== 多变量梯度 ===")
x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)
z = x * y + x ** 2

print(f"z = x*y + x^2 = {z}")
z.backward()
print(f"dz/dx = {x.grad}")
print(f"dz/dy = {y.grad}")

print("\n=== 禁用梯度追踪 ===")
with torch.no_grad():
    a = torch.tensor(3.0, requires_grad=True)
    b = a * 2
    print(f"b = {b}")
    print(f"b.requires_grad = {b.requires_grad}")

print("\n=== 复杂计算图 ===")
x = torch.linspace(0, 2*3.14159, 100, requires_grad=True)
y = torch.sin(x)
z = y.sum()
z.backward()

import matplotlib.pyplot as plt
plt.plot(x.detach().numpy(), y.detach().numpy(), label='sin(x)')
plt.plot(x.detach().numpy(), x.grad.numpy(), label='cos(x)')
plt.legend()
plt.title('sin(x) 及其导数 cos(x)')
plt.show()