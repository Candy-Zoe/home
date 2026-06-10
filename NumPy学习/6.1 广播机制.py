# NumPy广播机制学习
# 主要内容：广播规则、广播应用、高级广播技巧

import numpy as np

print("=== 基本广播 ===")
a = np.array([1, 2, 3])
b = np.array([2])

result = a * b
print(f"a = {a}")
print(f"b = {b}")
print(f"a * b = {result}")

print("\n=== 二维广播 ===")
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
v = np.array([1, 0, 1])

result = A + v
print(f"A:\n{A}")
print(f"v = {v}")
print(f"A + v:\n{result}")

print("\n=== 行向量广播 ===")
row = np.array([[1, 2, 3]])
col = np.array([[4], [5], [6]])

result = row + col
print(f"row:\n{row}")
print(f"col:\n{col}")
print(f"row + col:\n{result}")

print("\n=== 三维广播 ===")
A = np.ones((3, 4, 5))
B = np.ones((4, 5))

result = A + B
print(f"A.shape = {A.shape}")
print(f"B.shape = {B.shape}")
print(f"(A + B).shape = {result.shape}")

print("\n=== 广播应用：归一化 ===")
data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
mean = data.mean(axis=0)
std = data.std(axis=0)

normalized = (data - mean) / std
print(f"原始数据:\n{data}")
print(f"均值: {mean}")
print(f"标准差: {std}")
print(f"归一化后:\n{normalized}")

print("\n=== 广播应用：外积 ===")
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

outer = x[:, np.newaxis] * y[np.newaxis, :]
print(f"x = {x}")
print(f"y = {y}")
print(f"外积:\n{outer}")

print("\n=== 条件广播 ===")
A = np.array([[1, 2], [3, 4], [5, 6]])
mask = A > 3

print(f"A:\n{A}")
print(f"A > 3:\n{mask}")
print(f"A[A > 3] = {A[A > 3]}")

print("\n=== 高级广播技巧 ===")
A = np.arange(12).reshape(3, 4)
B = np.arange(4)

result = A[:, :, np.newaxis] * B[np.newaxis, :, np.newaxis]
print(f"A.shape = {A.shape}")
print(f"B.shape = {B.shape}")
print(f"result.shape = {result.shape}")