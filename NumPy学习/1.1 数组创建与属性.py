# NumPy数组创建与属性学习
# 主要内容：数组创建方法、数组属性、数据类型

import numpy as np

print("=== 创建数组 ===")

arr1 = np.array([1, 2, 3, 4])
print(f"一维数组: {arr1}")

arr2 = np.array([[1, 2], [3, 4], [5, 6]])
print(f"二维数组:\n{arr2}")

print("\n=== 特殊数组 ===")
zeros = np.zeros((2, 3))
print(f"零数组:\n{zeros}")

ones = np.ones((3, 3))
print(f"单位数组:\n{ones}")

eye = np.eye(4)
print(f"单位矩阵:\n{eye}")

print("\n=== 数组属性 ===")
print(f"数组形状: {arr2.shape}")
print(f"数组维度: {arr2.ndim}")
print(f"数组元素总数: {arr2.size}")
print(f"数组数据类型: {arr2.dtype}")

print("\n=== 数据类型 ===")
arr_int = np.array([1, 2, 3], dtype=np.int32)
print(f"int32数组: {arr_int}, 类型: {arr_int.dtype}")

arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print(f"float64数组: {arr_float}, 类型: {arr_float.dtype}")

arr_complex = np.array([1 + 2j, 3 + 4j])
print(f"复数数组: {arr_complex}, 类型: {arr_complex.dtype}")

print("\n=== 随机数组 ===")
np.random.seed(42)
rand_arr = np.random.rand(3, 4)
print(f"随机数组(0-1):\n{rand_arr}")

randn_arr = np.random.randn(3, 4)
print(f"标准正态分布数组:\n{randn_arr}")