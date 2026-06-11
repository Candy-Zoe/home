# NumPy文件操作学习
# 主要内容：数组保存、加载、CSV文件处理

# 导入NumPy库
import numpy as np

# 创建测试数组
print("=== 创建测试数组 ===")

# 创建一维数组
arr1d = np.array([1, 2, 3, 4, 5])
print(f"一维数组: {arr1d}")

# 创建二维数组
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"二维数组:\n{arr2d}")

# 创建包含不同数据类型的结构化数组
dt = np.dtype([('name', 'U10'), ('age', 'i4'), ('weight', 'f4')])
structured_arr = np.array([('Tom', 25, 60.5), ('Jerry', 30, 70.2)], dtype=dt)
print(f"\n结构化数组:\n{structured_arr}")

# 保存和加载数组
print("\n=== 保存和加载.npy文件 ===")

# 保存一维数组
np.save('arr1d.npy', arr1d)
print("一维数组已保存为 arr1d.npy")

# 加载一维数组
arr1d_loaded = np.load('arr1d.npy')
print(f"加载的一维数组: {arr1d_loaded}")

# 保存二维数组
np.save('arr2d.npy', arr2d)
print("\n二维数组已保存为 arr2d.npy")

# 加载二维数组
arr2d_loaded = np.load('arr2d.npy')
print(f"加载的二维数组:\n{arr2d_loaded}")

# 保存多个数组到一个文件
print("\n=== 保存多个数组 ===")

# 使用np.savez保存多个数组
np.savez('multiple_arrays.npz', array1=arr1d, array2=arr2d, name='test')
print("多个数组已保存为 multiple_arrays.npz")

# 加载多个数组
loaded = np.load('multiple_arrays.npz')
print(f"\n加载的数组名称: {list(loaded.keys())}")
print(f"array1: {loaded['array1']}")
print(f"array2:\n{loaded['array2']}")
print(f"name: {loaded['name']}")

# 使用np.savez_compressed压缩保存
np.savez_compressed('compressed_arrays.npz', array1=arr1d, array2=arr2d)
print("\n压缩数组已保存为 compressed_arrays.npz")

# 加载压缩文件
loaded_compressed = np.load('compressed_arrays.npz')
print(f"加载的压缩数组: {loaded_compressed['array1']}")

# CSV文件操作
print("\n=== CSV文件操作 ===")

# 保存为CSV文件
np.savetxt('array.csv', arr2d, delimiter=',', fmt='%d', header='列1,列2,列3')
print("二维数组已保存为 array.csv")

# 加载CSV文件
arr_csv = np.loadtxt('array.csv', delimiter=',', skiprows=1)
print(f"从CSV加载的数组:\n{arr_csv}")

# 保存为带格式的CSV
np.savetxt('formatted_array.csv', arr2d, delimiter=',', fmt='%.2f', 
           header='列1,列2,列3', comments='')
print("\n带格式的数组已保存为 formatted_array.csv")

# 加载带格式的CSV
arr_formatted = np.loadtxt('formatted_array.csv', delimiter=',', skiprows=1)
print(f"加载的格式数组:\n{arr_formatted}")

# 使用genfromtxt处理包含缺失值的数据
print("\n=== 处理缺失值 ===")

# 创建包含缺失值的示例数据
data_with_missing = """1,2,3
4,,6
7,8,9
,11,12"""

with open('missing_data.csv', 'w') as f:
    f.write(data_with_missing)

# 使用genfromtxt加载，自动处理缺失值
arr_missing = np.genfromtxt('missing_data.csv', delimiter=',')
print(f"包含缺失值的数组:\n{arr_missing}")

# 使用fill_value指定填充值
arr_filled = np.genfromtxt('missing_data.csv', delimiter=',', fill_value=0)
print(f"\n填充缺失值后的数组:\n{arr_filled}")

# 保存和加载文本文件
print("\n=== 文本文件操作 ===")

# 保存为文本文件
arr1d.tofile('array.txt', sep=',')
print("一维数组已保存为 array.txt")

# 从文本文件加载
arr_loaded = np.fromfile('array.txt', sep=',')
print(f"从文本文件加载的数组: {arr_loaded}")

# 处理大文件
print("\n=== 处理大文件 ===")

# 创建大数组
large_arr = np.arange(1000000)

# 保存大数组
np.save('large_array.npy', large_arr)
print(f"大数组已保存: {large_arr.nbytes / 1024 / 1024:.2f} MB")

# 内存映射文件（用于处理超大文件）
# 创建内存映射文件
fp = np.memmap('memmap_array.dat', dtype='float32', mode='w+', shape=(1000, 1000))
print(f"\n内存映射数组形状: {fp.shape}")

# 写入数据
fp[:100, :100] = np.random.rand(100, 100)

# 刷新到磁盘
fp.flush()

# 读取内存映射文件
fp_read = np.memmap('memmap_array.dat', dtype='float32', mode='r', shape=(1000, 1000))
print(f"读取内存映射数组前10x10:\n{fp_read[:10, :10]}")

# 结构化数组文件操作
print("\n=== 结构化数组文件操作 ===")

# 保存结构化数组
np.save('structured.npy', structured_arr)
print("结构化数组已保存")

# 加载结构化数组
struct_loaded = np.load('structured.npy', allow_pickle=True)
print(f"加载的结构化数组:\n{struct_loaded}")

# 保存为CSV
np.savetxt('structured.csv', structured_arr, delimiter=',', 
           fmt='%s,%d,%.1f', header='name,age,weight')
print("\n结构化数组已保存为CSV")

# 清理文件
print("\n=== 清理测试文件 ===")

import os
files_to_clean = [
    'arr1d.npy', 'arr2d.npy', 'multiple_arrays.npz', 
    'compressed_arrays.npz', 'array.csv', 'formatted_array.csv',
    'missing_data.csv', 'array.txt', 'large_array.npy',
    'memmap_array.dat', 'structured.npy', 'structured.csv'
]

for f in files_to_clean:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")

print("\n文件操作学习完成！")