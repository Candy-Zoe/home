# NumPy数组创建与属性学习
# 主要内容：数组创建方法、数组属性、数据类型

# 导入NumPy库，通常使用np作为别名
import numpy as np

# 创建数组的基本方法
print("=== 创建数组 ===")

# 使用列表创建一维数组
arr1 = np.array([1, 2, 3, 4])
print(f"一维数组: {arr1}")

# 使用嵌套列表创建二维数组（矩阵）
arr2 = np.array([[1, 2], [3, 4], [5, 6]])
print(f"二维数组:\n{arr2}")

# 创建特殊用途的数组
print("\n=== 特殊数组 ===")

# 创建全零数组，参数为形状元组
zeros = np.zeros((2, 3))
print(f"零数组:\n{zeros}")

# 创建全1数组
ones = np.ones((3, 3))
print(f"单位数组:\n{ones}")

# 创建单位矩阵（对角线为1，其余为0）
eye = np.eye(4)
print(f"单位矩阵:\n{eye}")

# 数组的基本属性
print("\n=== 数组属性 ===")

# shape: 返回数组的形状，即各维度的大小
print(f"数组形状: {arr2.shape}")

# ndim: 返回数组的维度数量
print(f"数组维度: {arr2.ndim}")

# size: 返回数组元素的总个数
print(f"数组元素总数: {arr2.size}")

# dtype: 返回数组的数据类型
print(f"数组数据类型: {arr2.dtype}")

# 数据类型说明
print("\n=== 数据类型 ===")

# 指定数据类型为int32（32位整数）
arr_int = np.array([1, 2, 3], dtype=np.int32)
print(f"int32数组: {arr_int}, 类型: {arr_int.dtype}")

# 指定数据类型为float64（64位浮点数）
arr_float = np.array([1.0, 2.0, 3.0], dtype=np.float64)
print(f"float64数组: {arr_float}, 类型: {arr_float.dtype}")

# NumPy支持复数类型
arr_complex = np.array([1 + 2j, 3 + 4j])
print(f"复数数组: {arr_complex}, 类型: {arr_complex.dtype}")

# 随机数组生成
print("\n=== 随机数组 ===")

# 设置随机种子，确保结果可重复
np.random.seed(42)

# 生成0-1之间均匀分布的随机数组
rand_arr = np.random.rand(3, 4)
print(f"随机数组(0-1):\n{rand_arr}")

# 生成标准正态分布（均值为0，标准差为1）的随机数组
randn_arr = np.random.randn(3, 4)
print(f"标准正态分布数组:\n{randn_arr}")