# NumPy矩阵操作学习
# 主要内容：矩阵创建、矩阵运算、线性代数操作

import numpy as np

print("=== 创建矩阵 ===")
mat1 = np.matrix([[1, 2], [3, 4]])
mat2 = np.matrix([[5, 6], [7, 8]])
print(f"矩阵mat1:\n{mat1}")
print(f"矩阵mat2:\n{mat2}")

print("\n=== 矩阵运算 ===")
print(f"矩阵加法:\n{mat1 + mat2}")
print(f"矩阵减法:\n{mat1 - mat2}")
print(f"矩阵乘法:\n{mat1 * mat2}")

print("\n=== 线性代数操作 ===")
print(f"矩阵转置:\n{mat1.T}")
print(f"矩阵逆:\n{mat1.I}")
print(f"矩阵行列式: {np.linalg.det(mat1)}")
print(f"矩阵秩: {np.linalg.matrix_rank(mat1)}")

print("\n=== 特征值与特征向量 ===")
eigenvalues, eigenvectors = np.linalg.eig(mat1)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

print("\n=== 求解线性方程组 ===")
A = np.array([[1, 2], [3, 4]])
b = np.array([5, 11])
x = np.linalg.solve(A, b)
print(f"方程组 Ax=b 的解: {x}")
print(f"验证: A@x = {A@x}")