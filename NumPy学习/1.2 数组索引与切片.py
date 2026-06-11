# NumPy数组索引与切片学习
# 主要内容：基本索引、花式索引、布尔索引、多维数组切片

# 导入NumPy库
import numpy as np

# 创建测试数组
print("=== 创建测试数组 ===")

# 创建一维数组
arr1d = np.arange(10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(f"一维数组: {arr1d}")

# 创建二维数组
arr2d = np.arange(12).reshape(3, 4)
print(f"二维数组:\n{arr2d}")

# 创建三维数组
arr3d = np.arange(24).reshape(2, 3, 4)
print(f"三维数组形状: {arr3d.shape}")

# 一维数组索引
print("\n=== 一维数组索引 ===")

# 正向索引（从0开始）
print(f"第0个元素: {arr1d[0]}")
print(f"第5个元素: {arr1d[5]}")

# 负向索引（从-1开始，即最后一个元素）
print(f"最后一个元素: {arr1d[-1]}")
print(f"倒数第二个元素: {arr1d[-2]}")

# 一维数组切片
print("\n=== 一维数组切片 ===")

# 起始:结束（不包括结束位置）
print(f"索引2到5: {arr1d[2:5]}")

# 省略起始（从0开始）
print(f"前5个元素: {arr1d[:5]}")

# 省略结束（到末尾）
print(f"索引5到末尾: {arr1d[5:]}")

# 省略起始和结束（复制整个数组）
print(f"整个数组: {arr1d[:]}")

# 步长切片
print(f"偶数索引元素: {arr1d[::2]}")  # 步长为2
print(f"奇数索引元素: {arr1d[1::2]}")  # 从索引1开始，步长为2
print(f"反向数组: {arr1d[::-1]}")  # 步长为-1，反向遍历

# 二维数组索引
print("\n=== 二维数组索引 ===")

# 通过行索引访问整行
print(f"第0行: {arr2d[0]}")

# 通过行索引和列索引访问单个元素
print(f"第1行第2列: {arr2d[1, 2]}")

# 通过方括号方式访问单个元素
print(f"第1行第2列(另一种方式): {arr2d[1][2]}")

# 二维数组切片
print("\n=== 二维数组切片 ===")

# 切片行
print(f"前2行:\n{arr2d[:2]}")

# 切片列
print(f"前2列:\n{arr2d[:, :2]}")

# 同时切片行和列（获取子矩阵）
print(f"前2行前2列:\n{arr2d[:2, :2]}")

# 获取特定行和列
print(f"第0行和第2行:\n{arr2d[[0, 2]]}")

# 获取特定列
print(f"第1列和第3列:\n{arr2d[:, [1, 3]]}")

# 花式索引
print("\n=== 花式索引 ===")

# 使用索引数组访问多个元素
indices = [0, 2, 4]
print(f"索引0,2,4的元素: {arr1d[indices]}")

# 使用二维索引数组
row_indices = [0, 1, 2]
col_indices = [0, 1, 2]
print(f"指定位置的元素: {arr2d[row_indices, col_indices]}")

# 使用布尔数组索引
print("\n=== 布尔索引 ===")

# 创建布尔条件
condition = arr1d > 5
print(f"条件 arr > 5: {condition}")
print(f"arr中大于5的元素: {arr1d[condition]}")

# 使用比较运算符创建条件
print(f"arr中等于3的元素: {arr1d[arr1d == 3]}")

# 组合多个条件
print(f"arr中大于2且小于8的元素: {arr1d[(arr1d > 2) & (arr1d < 8)]}")

# 使用np.where进行条件替换
result = np.where(arr1d > 5, arr1d, 0)
print(f"大于5保留，否则替换为0: {result}")

# 三维数组索引
print("\n=== 三维数组索引 ===")

print(f"第一个深度切片:\n{arr3d[0]}")

print(f"第一个深度切片的第0行: {arr3d[0, 0]}")

print(f"第一个深度切片第0行第0列: {arr3d[0, 0, 0]}")

# 三维数组切片
print("\n=== 三维数组切片 ===")

# 切片第一个维度
print(f"第一个深度:\n{arr3d[:1]}")

# 切片所有深度的前两行
print(f"所有深度的前两行:\n{arr3d[:, :2, :]}")

# 切片特定深度和行
print(f"特定深度和行:\n{arr3d[0, :, :2]}")

# 修改数组切片
print("\n=== 修改数组切片 ===")

# 创建一个数组
arr = np.arange(10)
print(f"原始数组: {arr}")

# 通过切片修改数组
arr[2:5] = 100
print(f"修改后 arr[2:5]=100: {arr}")

# 通过布尔索引修改数组
arr[arr > 50] = 200
print(f"大于50的元素改为200: {arr}")

# 使用np.copy创建副本
print("\n=== 副本与视图 ===")

arr = np.arange(10)
slice_copy = arr[2:5].copy()  # 创建副本
slice_view = arr[2:5]  # 创建视图（默认行为）

print(f"原始数组: {arr}")
print(f"切片副本: {slice_copy}")
print(f"切片视图: {slice_view}")

# 修改副本不会影响原数组
slice_copy[0] = 999
print(f"修改副本后: arr={arr}, 副本={slice_copy}")

# 修改视图会影响原数组
slice_view[0] = 888
print(f"修改视图后: arr={arr}, 视图={slice_view}")