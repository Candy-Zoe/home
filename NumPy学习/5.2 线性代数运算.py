# NumPy线性代数运算学习
# 主要内容：矩阵运算、特征值分解、奇异值分解、线性方程组求解

import numpy as np

print("=== 创建矩阵 ===")
A = np.array([[1, 2], [3, 4]], dtype=np.float64)
B = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")

print("\n=== 矩阵加法 ===")
C = A + B
print(f"A + B:\n{C}")

print("\n=== 矩阵乘法 ===")
D = A @ B
print(f"A @ B:\n{D}")

print("\n=== 矩阵转置 ===")
print(f"A的转置:\n{A.T}")

print("\n=== 矩阵逆 ===")
A_inv = np.linalg.inv(A)
print(f"A的逆:\n{A_inv}")

print("\n=== 矩阵行列式 ===")
det_A = np.linalg.det(A)
print(f"A的行列式: {det_A}")

print("\n=== 特征值和特征向量 ===")
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

print("\n=== 奇异值分解 ===")
U, S, V = np.linalg.svd(A)
print(f"U:\n{U}")
print(f"S: {S}")
print(f"V:\n{V}")

print("\n=== 线性方程组求解 ===")
b = np.array([1, 2])
x = np.linalg.solve(A, b)
print(f"Ax = b 的解: {x}")
print(f"验证: A @ x = {A @ x}")

print("\n=== 范数计算 ===")
vec = np.array([3, 4])
norm_l2 = np.linalg.norm(vec)
norm_l1 = np.linalg.norm(vec, ord=1)
print(f"L2范数: {norm_l2}")
print(f"L1范数: {norm_l1}")

print("\n=== 矩阵秩 ===")
rank = np.linalg.matrix_rank(A)
print(f"A的秩: {rank}")

print("\n=== 伪逆 ===")
pinv_A = np.linalg.pinv(A)
print(f"A的伪逆:\n{pinv_A}")