# NumPy文件读写学习
# 主要内容：保存和加载NumPy数组

import numpy as np

print("=== 保存和加载单个数组 ===")
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.save('example.npy', arr)
loaded_arr = np.load('example.npy')
print(f"原始数组:\n{arr}")
print(f"加载的数组:\n{loaded_arr}")

print("\n=== 保存和加载多个数组 ===")
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
np.savez('multiple_arrays.npz', arr1=arr1, arr2=arr2)
loaded = np.load('multiple_arrays.npz')
print(f"arr1: {loaded['arr1']}")
print(f"arr2: {loaded['arr2']}")

print("\n=== 保存为文本文件 ===")
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
np.savetxt('array.txt', arr, fmt='%d', delimiter=',')
loaded_txt = np.loadtxt('array.txt', delimiter=',')
print(f"文本文件加载:\n{loaded_txt}")

print("\n=== 清理测试文件 ===")
import os
for f in ['example.npy', 'multiple_arrays.npz', 'array.txt']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")