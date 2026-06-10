# NumPy数组形状操作学习
# 主要内容：reshape、转置、合并、分割

import numpy as np

print("=== reshape ===")
arr = np.arange(12)
print(f"原始数组: {arr}")
arr_3x4 = arr.reshape(3, 4)
print(f"3x4数组:\n{arr_3x4}")
arr_2x2x3 = arr.reshape(2, 2, 3)
print(f"2x2x3数组:\n{arr_2x2x3}")

print("\n=== 转置 ===")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"原始数组:\n{arr}")
print(f"转置:\n{arr.T}")
print(f"转置(另一种方式):\n{np.transpose(arr)}")

print("\n=== 合并数组 ===")
arr1 = np.array([[1, 2], [3, 4]])
arr2 = np.array([[5, 6], [7, 8]])

print(f"水平合并:\n{np.hstack((arr1, arr2))}")
print(f"垂直合并:\n{np.vstack((arr1, arr2))}")
print(f"concatenate水平:\n{np.concatenate((arr1, arr2), axis=1)}")
print(f"concatenate垂直:\n{np.concatenate((arr1, arr2), axis=0)}")

print("\n=== 分割数组 ===")
arr = np.array([1, 2, 3, 4, 5, 6])
print(f"原始数组: {arr}")
print(f"分成3份: {np.split(arr, 3)}")
print(f"按位置分割: {np.split(arr, [2, 4])}")

arr2d = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"原始二维数组:\n{arr2d}")
print(f"水平分割:\n{np.hsplit(arr2d, 2)}")
print(f"垂直分割:\n{np.vsplit(arr2d, 3)}")

print("\n=== flatten与ravel ===")
arr = np.array([[1, 2], [3, 4]])
print(f"原始数组:\n{arr}")
print(f"flatten: {arr.flatten()}")
print(f"ravel: {arr.ravel()}")