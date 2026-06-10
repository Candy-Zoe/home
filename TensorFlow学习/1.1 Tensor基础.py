# TensorFlow Tensor基础学习
# 主要内容：Tensor的创建、基本属性、操作

import tensorflow as tf

print("=== 创建Tensor ===")
scalar = tf.constant(5)
print(f"标量: {scalar}")

vector = tf.constant([1, 2, 3])
print(f"向量: {vector}")

matrix = tf.constant([[1, 2], [3, 4]])
print(f"矩阵:\n{matrix}")

print("\n=== Tensor属性 ===")
print(f"形状: {matrix.shape}")
print(f"数据类型: {matrix.dtype}")
print(f"维度: {matrix.ndim}")

print("\n=== 类型转换 ===")
float_tensor = tf.cast(matrix, tf.float32)
print(f"转换为float32:\n{float_tensor}")

print("\n=== Tensor运算 ===")
a = tf.constant([1, 2, 3])
b = tf.constant([4, 5, 6])

print(f"加法: {tf.add(a, b)}")
print(f"乘法: {tf.multiply(a, b)}")
print(f"矩阵乘法: {tf.matmul(matrix, matrix)}")

print("\n=== 特殊Tensor ===")
zeros = tf.zeros((2, 3))
print(f"零Tensor:\n{zeros}")

ones = tf.ones((3, 3))
print(f"单位Tensor:\n{ones}")

random = tf.random.normal((2, 2), mean=0, stddev=1)
print(f"随机Tensor:\n{random}")

print("\n=== 索引与切片 ===")
arr = tf.range(10)
print(f"原始Tensor: {arr}")
print(f"索引5: {arr[5]}")
print(f"切片1:4: {arr[1:4]}")

print("\n=== 形状操作 ===")
reshaped = tf.reshape(arr, (2, 5))
print(f"reshape(2,5):\n{reshaped}")

transposed = tf.transpose(matrix)
print(f"转置:\n{transposed}")