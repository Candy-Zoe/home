# PyTorch张量基础操作学习
# 主要内容：张量的创建、基本属性、索引切片、数学运算

import torch

# 创建张量
print("=== 创建张量 ===")
x = torch.tensor([1, 2, 3])
print(f"一维张量: {x}")

y = torch.tensor([[1, 2], [3, 4]])
print(f"二维张量:\n{y}")

# 张量属性
print("\n=== 张量属性 ===")
print(f"形状: {y.shape}")
print(f"数据类型: {y.dtype}")
print(f"设备: {y.device}")

# 索引和切片
print("\n=== 索引和切片 ===")
print(f"第一行: {y[0]}")
print(f"第二列: {y[:, 1]}")

# 数学运算
print("\n=== 数学运算 ===")
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
print(f"加法: {a + b}")
print(f"乘法: {a * b}")
print(f"矩阵乘法: {torch.matmul(y, y)}")

# 特殊张量
print("\n=== 特殊张量 ===")
zeros = torch.zeros(2, 3)
print(f"零张量:\n{zeros}")

ones = torch.ones(3, 3)
print(f"单位张量:\n{ones}")

rand = torch.rand(2, 2)
print(f"随机张量:\n{rand}")