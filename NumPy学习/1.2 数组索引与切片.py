# NumPy数组索引与切片学习
# 主要内容：基本索引、切片操作、布尔索引、花式索引

import numpy as np

print("=== 一维数组索引 ===")
arr = np.array([0, 1, 2, 3, 4, 5])
print(f"原始数组: {arr}")
print(f"索引0: {arr[0]}")
print(f"索引-1: {arr[-1]}")
print(f"切片1:4: {arr[1:4]}")
print(f"步长为2: {arr[::2]}")

print("\n=== 二维数组索引 ===")
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原始数组:\n{arr2d}")
print(f"第二行: {arr2d[1]}")
print(f"第一行第二列: {arr2d[0, 1]}")
print(f"前两行: \n{arr2d[:2]}")
print(f"所有行的前两列: \n{arr2d[:, :2]}")

print("\n=== 布尔索引 ===")
arr = np.array([1, 2, 3, 4, 5, 6])
mask = arr > 3
print(f"原始数组: {arr}")
print(f"大于3的元素: {arr[mask]}")

arr2d = np.array([[1, 2], [3, 4], [5, 6]])
mask2d = arr2d[:, 0] > 2
print(f"第一列大于2的行:\n{arr2d[mask2d]}")

print("\n=== 花式索引 ===")
arr = np.array([10, 20, 30, 40, 50])
indices = [1, 3, 0]
print(f"原始数组: {arr}")
print(f"花式索引[1,3,0]: {arr[indices]}")

arr2d = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
rows = [0, 2]
cols = [1, 0]
print(f"原始数组:\n{arr2d}")
print(f"花式索引({rows}, {cols}): {arr2d[rows, cols]}")