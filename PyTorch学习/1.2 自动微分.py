# PyTorch自动微分学习
# 主要内容：梯度计算、自动微分机制、自定义梯度

# 导入PyTorch库
import torch

# 自动微分基础
print("=== 自动微分基础 ===")

# 创建需要梯度的张量
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"张量x: {x}")
print(f"是否需要梯度: {x.requires_grad}")

# 定义计算图
y = x ** 2 + 2 * x + 1  # y = x^2 + 2x + 1
print(f"计算结果y: {y}")

# 计算梯度
y.sum().backward()  # 对y求和后再反向传播
print(f"梯度dy/dx: {x.grad}")

# 手动验证：d(x^2 + 2x + 1)/dx = 2x + 2
print(f"理论梯度(2x + 2): {2 * x + 2}")

# 梯度计算示例
print("\n=== 梯度计算示例 ===")

# 示例1：复杂函数
w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

x = torch.tensor([2.0, 3.0, 4.0])
y = w * x + b  # 线性函数 y = wx + b

loss = y.sum()
loss.backward()

print(f"权重w的梯度: {w.grad}")
print(f"偏置b的梯度: {b.grad}")

# 示例2：矩阵乘法
print("\n=== 矩阵乘法梯度 ===")

A = torch.randn(3, 4, requires_grad=True)
B = torch.randn(4, 2, requires_grad=True)

C = A @ B
loss = C.sum()
loss.backward()

print(f"A的梯度形状: {A.grad.shape}")
print(f"B的梯度形状: {B.grad.shape}")

# 梯度清零与累积
print("\n=== 梯度清零与累积 ===")

x = torch.tensor(2.0, requires_grad=True)

# 第一次反向传播
y = x ** 2
y.backward()
print(f"第一次梯度: {x.grad}")

# 清零梯度
x.grad.zero_()
print(f"清零后: {x.grad}")

# 第二次反向传播
y = x ** 3
y.backward()
print(f"第二次梯度: {x.grad}")

# 梯度累积示例
print("\n=== 梯度累积 ===")

x = torch.tensor(1.0, requires_grad=True)

# 模拟多次前向传播累积梯度
for i in range(3):
    y = x ** 2
    y.backward()
    print(f"第{i+1}次迭代梯度: {x.grad}")

print(f"累积后的总梯度: {x.grad}")

# 使用no_grad停止追踪
print("\n=== no_grad上下文 ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2

print(f"在no_grad外: y.requires_grad = {y.requires_grad}")

with torch.no_grad():
    z = x * 3
    print(f"在no_grad内: z.requires_grad = {z.requires_grad}")

# 使用detach分离张量
print("\n=== detach分离张量 ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2

# 分离y，使其不需要梯度
y_detached = y.detach()
print(f"分离后的y: {y_detached}")
print(f"分离后y.requires_grad: {y_detached.requires_grad}")

# 自定义函数（使用torch.autograd.Function）
print("\n=== 自定义函数 ===")

class PowerFunction(torch.autograd.Function):
    """自定义幂函数及其梯度计算"""

    @staticmethod
    def forward(ctx, input, power):
        """前向传播"""
        ctx.save_for_backward(input, power)
        return input ** power

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播：计算梯度"""
        input, power = ctx.saved_tensors
        return grad_output * power * (input ** (power - 1)), None

# 使用自定义函数
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
power = torch.tensor(2.0)

y = PowerFunction.apply(x, power)
y.sum().backward()

print(f"输入x: {x}")
print(f"输出y: {y}")
print(f"梯度dy/dx: {x.grad}")

# 计算图的可视化
print("\n=== 计算图属性 ===")

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = y.sum()
w = z * 2

print(f"x.grad_fn: {x.grad_fn}")  # None，x是叶子节点
print(f"y.grad_fn: {y.grad_fn}")  # PowBackward
print(f"z.grad_fn: {z.grad_fn}")  # SumBackward
print(f"w.grad_fn: {w.grad_fn}")  # MulBackward

# 检查叶子节点
print(f"x是叶子节点: {x.is_leaf}")
print(f"y是叶子节点: {y.is_leaf}")

# 高阶导数
print("\n=== 高阶导数 ===")

x = torch.tensor(2.0, requires_grad=True)

# 一阶导数：dy/dx = 2x
y = x ** 2
first_grad = torch.autograd.grad(y, x, create_graph=True)[0]
print(f"一阶导数 dy/dx = {first_grad}")

# 二阶导数：d²y/dx² = 2
second_grad = torch.autograd.grad(first_grad, x)[0]
print(f"二阶导数 d²y/dx² = {second_grad}")

# 梯度模式
print("\n=== 梯度模式 ===")

# 设置整个网络的梯度模式
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)

# 默认使用反向模式微分
with torch.set_grad_enabled(True):
    y1 = x ** 2
    print(f"启用梯度: y1.requires_grad = {y1.requires_grad}")

with torch.set_grad_enabled(False):
    y2 = x ** 2
    print(f"禁用梯度: y2.requires_grad = {y2.requires_grad}")