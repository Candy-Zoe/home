# NumPy数组统计与聚合学习
# 主要内容：统计函数、聚合操作、排序

import numpy as np

print("=== 基本统计函数 ===")
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"数组: {arr}")
print(f"求和: {np.sum(arr)}")
print(f"均值: {np.mean(arr)}")
print(f"中位数: {np.median(arr)}")
print(f"标准差: {np.std(arr)}")
print(f"方差: {np.var(arr)}")
print(f"最小值: {np.min(arr)}")
print(f"最大值: {np.max(arr)}")
print(f"最小值索引: {np.argmin(arr)}")
print(f"最大值索引: {np.argmax(arr)}")

print("\n=== 二维数组统计 ===")
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原始数组:\n{arr2d}")
print(f"全部求和: {np.sum(arr2d)}")
print(f"按列求和: {np.sum(arr2d, axis=0)}")
print(f"按行求和: {np.sum(arr2d, axis=1)}")
print(f"按列求均值: {np.mean(arr2d, axis=0)}")
print(f"按行求均值: {np.mean(arr2d, axis=1)}")

print("\n=== 排序 ===")
arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"原始数组: {arr}")
sorted_arr = np.sort(arr)
print(f"排序后: {sorted_arr}")

arr2d = np.array([[3, 1], [4, 2]])
print(f"原始二维数组:\n{arr2d}")
print(f"按行排序:\n{np.sort(arr2d, axis=1)}")
print(f"按列排序:\n{np.sort(arr2d, axis=0)}")

print("\n=== 唯一值与计数 ===")
arr = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print(f"原始数组: {arr}")
print(f"唯一值: {np.unique(arr)}")
values, counts = np.unique(arr, return_counts=True)
print(f"唯一值及其计数: {dict(zip(values, counts))}")

print("\n=== 其他聚合函数 ===")
arr = np.array([1, 2, 3, 4, 5])
print(f"累积和: {np.cumsum(arr)}")
print(f"累积积: {np.cumprod(arr)}")
print(f"乘积: {np.prod(arr)}")