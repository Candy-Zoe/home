# PyTorch张量基础学习
# 主要内容：张量的创建、基本属性、索引切片、数学运算

# 导入PyTorch库
import torch
import numpy as np

# 创建张量的方法
print("=== 创建张量 ===")

# 从列表创建张量
tensor1 = torch.tensor([1, 2, 3, 4])
print(f"从列表创建的张量: {tensor1}")

# 从NumPy数组创建张量
np_array = np.array([[1, 2], [3, 4]])
tensor2 = torch.from_numpy(np_array)
print(f"从NumPy数组创建的张量:\n{tensor2}")

# 创建全零张量
zeros = torch.zeros(3, 4)
print(f"全零张量:\n{zeros}")

# 创建全1张量
ones = torch.ones(2, 3)
print(f"全1张量:\n{ones}")

# 创建随机张量（均匀分布，0-1之间）
rand_tensor = torch.rand(2, 3)
print(f"随机张量(0-1):\n{rand_tensor}")

# 创建标准正态分布的随机张量
randn_tensor = torch.randn(2, 3)
print(f"标准正态分布张量:\n{randn_tensor}")

# 张量的基本属性
print("\n=== 张量属性 ===")

# shape: 张量的形状
print(f"张量形状: {tensor2.shape}")

# dtype: 数据类型
print(f"数据类型: {tensor2.dtype}")

# device: 张量所在的设备（CPU或GPU）
print(f"所在设备: {tensor2.device}")

# requires_grad: 是否需要计算梯度
print(f"是否需要梯度: {tensor2.requires_grad}")

# 张量操作
print("\n=== 张量操作 ===")

# 加法运算
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b
print(f"张量加法: {a} + {b} = {c}")

# 乘法运算（逐元素相乘）
d = a * b
print(f"张量乘法: {a} * {b} = {d}")

# 矩阵乘法
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 4)
mat_result = mat1 @ mat2
print(f"矩阵乘法结果形状: {mat_result.shape}")

# 索引和切片
print("\n=== 索引和切片 ===")

tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原始张量:\n{tensor}")

# 访问第0行
print(f"第0行: {tensor[0]}")

# 访问第0行第1列的元素
print(f"第0行第1列: {tensor[0, 1]}")

# 访问前2行
print(f"前2行:\n{tensor[:2]}")

# 访问所有行的第0列
print(f"所有行的第0列: {tensor[:, 0]}")

# 张量变形
print("\n=== 张量变形 ===")

# reshape: 改变张量形状，元素总数不变
reshaped = tensor.reshape(1, 9)
print(f"变形为1x9:\n{reshaped}")

# view: 与reshape类似，但要求内存连续
viewed = tensor.view(9, 1)
print(f"view为9x1:\n{viewed}")

# 转置
transposed = tensor.t()
print(f"转置:\n{transposed}")

# 广播机制
print("\n=== 广播机制 ===")

x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([10, 20, 30])
result = x + y
print(f"原始张量x:\n{x}")
print(f"张量y: {y}")
print(f"广播相加结果:\n{result}")

# 设备切换
print("\n=== 设备切换 ===")

# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    tensor_gpu = tensor.to(device)
    print(f"张量已移动到GPU: {tensor_gpu.device}")
else:
    print("没有可用的GPU，使用CPU")