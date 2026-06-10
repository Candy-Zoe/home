# NumPy数组运算学习
# 主要内容：算术运算、广播机制、矩阵运算

import numpy as np

print("=== 基本算术运算 ===")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(f"加法: {arr1 + arr2}")
print(f"减法: {arr1 - arr2}")
print(f"乘法: {arr1 * arr2}")
print(f"除法: {arr1 / arr2}")
print(f"幂运算: {arr1 ** 2}")

print("\n=== 标量运算 ===")
arr = np.array([1, 2, 3])
print(f"数组+1: {arr + 1}")
print(f"数组*2: {arr * 2}")
print(f"数组/2: {arr / 2}")

print("\n=== 广播机制 ===")
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([10, 20, 30])
print(f"二维数组 + 一维数组:\n{arr1 + arr2}")

arr3 = np.array([[1], [2], [3]])
arr4 = np.array([10, 20])
print(f"形状(3,1) + 形状(2,):\n{arr3 + arr4}")

print("\n=== 矩阵运算 ===")
mat1 = np.array([[1, 2], [3, 4]])
mat2 = np.array([[5, 6], [7, 8]])

print(f"逐元素乘法:\n{mat1 * mat2}")
print(f"矩阵乘法:\n{np.dot(mat1, mat2)}")
print(f"矩阵乘法(另一种方式):\n{mat1 @ mat2}")

print("\n=== 数学函数 ===")
arr = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"原始数组(弧度): {arr}")
print(f"sin: {np.sin(arr)}")
print(f"cos: {np.cos(arr)}")
print(f"exp: {np.exp(arr)}")
print(f"log: {np.log(arr + 1)}")