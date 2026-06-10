# NumPy高级技巧与性能优化学习
# 主要内容：结构化数组、内存布局、性能技巧、并行计算

import numpy as np
import time

print("=== 结构化数组 ===")
dt = np.dtype([('name', 'U20'), ('age', 'i4'), ('weight', 'f4')])
people = np.array([('Alice', 25, 55.5), ('Bob', 30, 80.0)], dtype=dt)

print(f"结构化数组:\n{people}")
print(f"访问字段: {people['name']}")
print(f"年龄: {people['age']}")

print("\n=== 记录数组 ===")
rec = np.rec.array([(1, 2.0), (3, 4.0)], dtype=[('x', 'i4'), ('y', 'f4')])
print(f"记录数组:\n{rec}")
print(f"x字段: {rec.x}")
print(f"y字段: {rec.y}")

print("\n=== 内存布局 ===")
x = np.arange(100).reshape(10, 10)
print(f"C连续: {x.flags['C_CONTIGUOUS']}")
print(f"F连续: {x.flags['F_CONTIGUOUS']}")

y = np.asfortranarray(x)
print(f"转置后F连续: {y.flags['F_CONTIGUOUS']}")

print("\n=== 内存视图 ===")
x = np.arange(10)
y = x[::2]
print(f"视图共享数据: {np.shares_memory(x, y)}")

z = y.copy()
print(f"复制后不共享: {np.shares_memory(y, z)}")

print("\n=== 内存优化 ===")
x = np.arange(1000000, dtype=np.int32)
y = np.arange(1000000, dtype=np.int32)

start = time.time()
result = x + y
print(f"加法耗时: {time.time() - start:.6f}s")

print("\n=== Strides操作 ===")
x = np.arange(12).reshape(3, 4)
print(f"原始形状: {x.shape}")
print(f"步长: {x.strides}")

y = np.lib.stride_tricks.as_strided(x, shape=(2, 3), strides=(16, 8))
print(f"滑动窗口形状: {y.shape}")
print(f"滑动窗口:\n{y}")

print("\n=== 广播规则详解 ===")
a = np.arange(12).reshape(3, 4)
b = np.array([1, 2, 3, 4])

print(f"a.shape: {a.shape}")
print(f"b.shape: {b.shape}")
print(f"广播结果: {(a + b).shape}")
print(f"a + b:\n{a + b}")

print("\n=== where函数 ===")
a = np.arange(10)
condition = a > 5
result = np.where(condition, a**2, a)
print(f"条件结果: {result}")

print("\n=== piecewise函数 ===")
x = np.arange(10)
result = np.piecewise(x, [x < 3, x > 7], [lambda x: x**2, lambda x: x*2])
print(f"分段结果: {result}")

print("\n=== searchsorted ===")
a = np.array([1, 2, 3, 4, 5])
indices = np.searchsorted(a, [0.5, 2.5, 6])
print(f"插入位置: {indices}")

print("\n=== 累积函数 ===")
a = np.array([1, 2, 3, 4, 5])
print(f"累积和: {np.cumsum(a)}")
print(f"累积积: {np.cumproduct(a)}")
print(f"累积最小: {np.fmin.accumulate(a)}")

print("\n=== 数组拼接技巧 ===")
a = np.arange(6).reshape(2, 3)
b = np.arange(6, 12).reshape(2, 3)

c = np.concatenate([a, b], axis=0)
print(f"垂直拼接形状: {c.shape}")

d = np.concatenate([a, b], axis=1)
print(f"水平拼接形状: {d.shape}")

print("\n=== 稀疏矩阵转换 ===")
dense = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
sparse = np.where(dense != 0)
values = dense[sparse]
print(f"稀疏表示: 索引={sparse}, 值={values}")

print("\n=== 性能基准测试 ===")
n = 10000
x = np.random.randn(n, n)

start = time.time()
result = np.dot(x, x)
print(f"矩阵乘法耗时: {time.time() - start:.4f}s")

start = time.time()
result = np.sum(x)
print(f"求和耗时: {time.time() - start:.4f}s")