# TensorFlow Tensor基础学习
# 主要内容：Tensor的创建、基本属性、操作

# 导入TensorFlow库，通常使用tf作为别名
import tensorflow as tf
import numpy as np

# 创建Tensor的方法
print("=== 创建Tensor ===")

# 从列表创建Tensor
tensor1 = tf.constant([1, 2, 3, 4])
print(f"从列表创建的Tensor: {tensor1}")

# 从NumPy数组创建Tensor
np_array = np.array([[1, 2], [3, 4]])
tensor2 = tf.convert_to_tensor(np_array)
print(f"从NumPy数组创建的Tensor:\n{tensor2}")

# 创建全零Tensor
zeros = tf.zeros((2, 3))
print(f"全零Tensor:\n{zeros}")

# 创建全1Tensor
ones = tf.ones((3, 3))
print(f"全1Tensor:\n{ones}")

# 创建随机Tensor（均匀分布）
rand_tensor = tf.random.uniform((2, 3), minval=0, maxval=1)
print(f"均匀分布随机Tensor:\n{rand_tensor}")

# 创建正态分布的随机Tensor
randn_tensor = tf.random.normal((2, 3), mean=0, stddev=1)
print(f"正态分布随机Tensor:\n{randn_tensor}")

# Tensor的基本属性
print("\n=== Tensor属性 ===")

# shape: Tensor的形状
print(f"Tensor形状: {tensor2.shape}")

# dtype: 数据类型
print(f"数据类型: {tensor2.dtype}")

# rank: 张量的阶数（维度数量）
print(f"张量阶数: {tensor2.rank()}")

# Tensor操作
print("\n=== Tensor操作 ===")

# 加法运算
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])
c = tf.add(a, b)
print(f"Tensor加法: {a} + {b} = {c}")

# 乘法运算（逐元素相乘）
d = tf.multiply(a, b)
print(f"Tensor乘法: {a} * {b} = {d}")

# 矩阵乘法
mat1 = tf.random.normal((2, 3))
mat2 = tf.random.normal((3, 4))
mat_result = tf.matmul(mat1, mat2)
print(f"矩阵乘法结果形状: {mat_result.shape}")

# 索引和切片
print("\n=== 索引和切片 ===")

tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"原始Tensor:\n{tensor}")

# 访问第0行
print(f"第0行: {tensor[0]}")

# 访问第0行第1列的元素
print(f"第0行第1列: {tensor[0, 1]}")

# 访问前2行
print(f"前2行:\n{tensor[:2]}")

# 访问所有行的第0列
print(f"所有行的第0列: {tensor[:, 0]}")

# Tensor变形
print("\n=== Tensor变形 ===")

# reshape: 改变Tensor形状
reshaped = tf.reshape(tensor, (1, 9))
print(f"变形为1x9:\n{reshaped}")

# 转置
transposed = tf.transpose(tensor)
print(f"转置:\n{transposed}")

# 数学运算
print("\n=== 数学运算 ===")

x = tf.constant([1.0, 2.0, 3.0])

# 平方
square = tf.square(x)
print(f"平方: {square}")

# 平方根
sqrt = tf.sqrt(x)
print(f"平方根: {sqrt}")

# 指数
exp = tf.exp(x)
print(f"指数: {exp}")

# 对数
log = tf.math.log(x)
print(f"自然对数: {log}")

# 设备切换
print("\n=== 设备切换 ===")

# 检查是否有可用的GPU
if tf.config.list_physical_devices('GPU'):
    tensor_gpu = tensor.gpu()
    print(f"Tensor已移动到GPU")
else:
    print("没有可用的GPU，使用CPU")