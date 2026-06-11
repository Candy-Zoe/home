# NumPy高级数值计算学习
# 主要内容：广播机制、向量化、高级索引、线性代数、随机数生成、性能优化

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import time

# 广播机制
print("=== 广播机制 ===")

# 基本广播示例
print("1. 基本广播 - 标量和数组:")
a = np.array([1, 2, 3])
b = 2
result = a + b
print(f"  {a} + {b} = {result}")

# 不同形状数组的广播
print("\n2. 不同形状数组的广播:")
a = np.array([[1, 2, 3], [4, 5, 6]])  # shape: (2, 3)
b = np.array([10, 20, 30])  # shape: (3,)
result = a + b
print(f"  a shape: {a.shape}, b shape: {b.shape}")
print(f"  结果:\n{result}")

# 广播规则示例
print("\n3. 广播规则:")
x = np.array([1, 2, 3])  # shape: (3,)
y = np.array([[1], [2], [3]])  # shape: (3, 1)
result = x + y
print(f"  x shape: {x.shape}")
print(f"  y shape: {y.shape}")
print(f"  x + y:\n{result}")
print(f"  结果 shape: {result.shape}")

# 无法广播的情况
print("\n4. 不兼容的形状 (尝试广播时会出错):")
a = np.array([1, 2, 3])  # shape: (3,)
b = np.array([1, 2])  # shape: (2,)
try:
    result = a + b
except ValueError as e:
    print(f"  错误: {e}")

# 广播应用 - 标准化
print("\n5. 广播应用 - 矩阵标准化:")
matrix = np.random.randn(5, 4)  # 5个样本，4个特征
mean = matrix.mean(axis=0)  # shape: (4,)
std = matrix.std(axis=0)  # shape: (4,)
standardized = (matrix - mean) / std
print(f"  原始矩阵形状: {matrix.shape}")
print(f"  均值形状: {mean.shape}")
print(f"  标准差形状: {std.shape}")
print(f"  标准化后均值: {standardized.mean(axis=0).round(4)}")
print(f"  标准化后标准差: {standardized.std(axis=0).round(4)}")

# 向量化操作
print("\n=== 向量化操作 ===")

# 创建示例数据
size = 1000000
a = np.random.rand(size)
b = np.random.rand(size)

# 比较循环和向量化的速度
print("1. 循环vs向量化速度对比:")

# Python循环
start = time.time()
result_loop = [x + y for x, y in zip(a, b)]
end = time.time()
loop_time = end - start

# NumPy向量化
start = time.time()
result_vec = a + b
end = time.time()
vec_time = end - start

print(f"  Python循环: {loop_time:.4f} 秒")
print(f"  NumPy向量化: {vec_time:.4f} 秒")
print(f"  速度提升: {loop_time / vec_time:.2f}x")

# 向量化函数
print("\n2. 向量化函数:")

# 定义一个函数
def my_function(x):
    if x > 0.5:
        return x ** 2
    else:
        return x ** 3

# 使用np.vectorize
vectorized_func = np.vectorize(my_function)

# 测试
arr = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
result = vectorized_func(arr)
print(f"  输入: {arr}")
print(f"  输出: {result}")

# 使用np.where (更高效)
result_where = np.where(arr > 0.5, arr ** 2, arr ** 3)
print(f"  np.where结果: {result_where}")

# 高级索引
print("\n=== 高级索引 ===")

# 创建示例数组
arr = np.array([[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]])

print("1. 整数数组索引:")
row_indices = np.array([0, 1, 2])
col_indices = np.array([1, 2, 0])
result = arr[row_indices, col_indices]
print(f"  选择 arr[{row_indices}, {col_indices}] = {result}")

# 布尔索引
print("\n2. 布尔索引:")
mask = arr > 5
print(f"  大于5的元素位置:\n{mask}")
print(f"  大于5的元素: {arr[mask]}")

# 组合条件
print("\n3. 组合条件索引:")
mask = (arr > 3) & (arr < 9)
print(f"  3<x<9的元素位置:\n{mask}")
print(f"  3<x<9的元素: {arr[mask]}")

# 花式索引
print("\n4. 花式索引 (Fancy Indexing):")
arr_2d = np.random.randint(0, 100, (10, 10))
rows = [0, 3, 5, 7]
cols = [1, 2, 8, 9]
result = arr_2d[rows][:, cols]
print(f"  选择行 {rows} 和列 {cols}:")
print(result)

# 对角线和上/下三角
print("\n5. 特殊索引:")
print(f"  对角线元素: {np.diag(arr)}")
print(f"  上三角 (k=1):\n{np.triu(arr, k=1)}")
print(f"  下三角 (k=-1):\n{np.tril(arr, k=-1)}")

# 线性代数
print("\n=== 线性代数 ===")

# 创建矩阵
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])
B = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
x = np.array([1, 2, 3])

# 矩阵乘法
print("1. 矩阵乘法:")
print(f"  A = \n{A}")
print(f"  B = \n{B}")
print(f"  A @ B = \n{np.dot(A, B)}")  # 或者 A @ B

# 矩阵属性
print("\n2. 矩阵属性:")
print(f"  转置 A.T:\n{A.T}")
print(f"  迹 (对角线元素和): {np.trace(A)}")
print(f"  行列式: {np.linalg.det(A):.2f}")
try:
    print(f"  逆矩阵:\n{np.linalg.inv(A[:2, :2])}")
except np.linalg.LinAlgError:
    print(f"  矩阵不可逆")

# 特征值和特征向量
print("\n3. 特征值和特征向量:")
eigenvalues, eigenvectors = np.linalg.eig(A[:2, :2])
print(f"  特征值: {eigenvalues}")
print(f"  特征向量:\n{eigenvectors}")

# 奇异值分解 (SVD)
print("\n4. 奇异值分解 (SVD):")
U, S, V = np.linalg.svd(A)
print(f"  U形状: {U.shape}")
print(f"  S: {S}")
print(f"  V形状: {V.shape}")

# 求解线性方程组
print("\n5. 求解线性方程组 Ax = b:")
A = np.array([[2, 1], [1, 1]])
b = np.array([5, 3])
x = np.linalg.solve(A, b)
print(f"  A = \n{A}")
print(f"  b = {b}")
print(f"  解 x = {x}")
print(f"  验证 Ax = {A @ x}")

# 范数计算
print("\n6. 范数计算:")
vec = np.array([3, 4])
print(f"  向量: {vec}")
print(f"  L1范数 (曼哈顿距离): {np.linalg.norm(vec, 1)}")
print(f"  L2范数 (欧几里得距离): {np.linalg.norm(vec, 2)}")
print(f"  无穷范数: {np.linalg.norm(vec, np.inf)}")

# 随机数生成
print("\n=== 随机数生成 ===")

# 设置随机种子
np.random.seed(42)
print("1. 设置随机种子 (seed=42)")

# 均匀分布
print("\n2. 均匀分布:")
uniform = np.random.uniform(low=0, high=1, size=5)
print(f"  均匀分布 [0,1]: {uniform.round(4)}")

# 正态分布
print("\n3. 正态分布:")
normal = np.random.normal(loc=0, scale=1, size=5)
print(f"  标准正态分布 N(0,1): {normal.round(4)}")
normal2 = np.random.normal(loc=5, scale=2, size=5)
print(f"  正态分布 N(5, 2): {normal2.round(4)}")

# 二项分布
print("\n4. 二项分布:")
binomial = np.random.binomial(n=10, p=0.5, size=5)
print(f"  二项分布 B(10, 0.5): {binomial}")

# 随机整数
print("\n5. 随机整数:")
randint = np.random.randint(low=0, high=100, size=10)
print(f"  随机整数 [0,100): {randint}")

# 随机排列
print("\n6. 随机排列:")
arr = np.arange(10)
print(f"  原始数组: {arr}")
shuffled = np.random.permutation(arr)
print(f"  随机排列: {shuffled}")

# 随机选择
print("\n7. 随机选择:")
choices = np.random.choice([1, 2, 3, 4, 5], size=10, p=[0.1, 0.2, 0.3, 0.2, 0.2])
print(f"  加权随机选择: {choices}")

# 随机数可视化
print("\n8. 随机数分布可视化:")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].hist(np.random.rand(10000), bins=30, edgecolor='black', alpha=0.7)
axes[0, 0].set_title('均匀分布')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(np.random.randn(10000), bins=30, edgecolor='black', alpha=0.7)
axes[0, 1].set_title('标准正态分布')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].hist(np.random.binomial(10, 0.5, 10000), bins=10, edgecolor='black', alpha=0.7)
axes[1, 0].set_title('二项分布 B(10, 0.5)')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].hist(np.random.exponential(scale=2, size=10000), bins=30, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('指数分布')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 数组重塑和变换
print("\n=== 数组重塑和变换 ===")

# 创建数组
arr = np.arange(24)
print(f"1. 原始数组: {arr}")
print(f"   形状: {arr.shape}")

# reshape
print("\n2. Reshape:")
arr_2d = arr.reshape(4, 6)
print(f"   4x6:\n{arr_2d}")
print(f"   形状: {arr_2d.shape}")

arr_3d = arr.reshape(2, 3, 4)
print(f"   2x3x4 形状: {arr_3d.shape}")

# flatten vs ravel
print("\n3. Flatten vs Ravel:")
flattened = arr_2d.flatten()  # 创建副本
raveled = arr_2d.ravel()  # 返回视图（可能）
print(f"   Flatten: {flattened}")
print(f"   Ravel: {raveled}")

# 转置
print("\n4. 转置:")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"   原始:\n{arr}")
print(f"   转置:\n{arr.T}")

# 交换轴
print("\n5. 交换轴 (swapaxes):")
arr = np.random.rand(2, 3, 4)
print(f"   原始形状: {arr.shape}")
swapped = np.swapaxes(arr, 0, 1)
print(f"   交换轴0和1后: {swapped.shape}")

# 连接和拆分数组
print("\n=== 连接和拆分数组 ===")

# 创建示例数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

print("1. 水平连接 (hstack):")
hstacked = np.hstack([a, b])
print(f"   结果:\n{hstacked}")
print(f"   形状: {hstacked.shape}")

print("\n2. 垂直连接 (vstack):")
vstacked = np.vstack([a, b])
print(f"   结果:\n{vstacked}")
print(f"   形状: {vstacked.shape}")

print("\n3. 沿指定轴连接 (concatenate):")
concat_0 = np.concatenate([a, b], axis=0)
concat_1 = np.concatenate([a, b], axis=1)
print(f"   axis=0:\n{concat_0}")
print(f"   axis=1:\n{concat_1}")

# 拆分数组
print("\n4. 水平拆分 (hsplit):")
arr = np.arange(16).reshape(4, 4)
split_arr = np.hsplit(arr, 2)
print(f"   原始:\n{arr}")
print(f"   拆分后第一个:\n{split_arr[0]}")
print(f"   拆分后第二个:\n{split_arr[1]}")

# 通用函数 (ufunc)
print("\n=== 通用函数 (ufunc) ===")

# 创建示例数据
x = np.array([0, np.pi/2, np.pi])
print("1. 三角函数:")
print(f"   x = {x}")
print(f"   sin(x) = {np.sin(x).round(4)}")
print(f"   cos(x) = {np.cos(x).round(4)}")
print(f"   tan(x) = {np.tan(x).round(4)}")

# 指数和对数
print("\n2. 指数和对数:")
x = np.array([1, 2, 3])
print(f"   x = {x}")
print(f"   e^x = {np.exp(x).round(4)}")
print(f"   2^x = {np.power(2, x)}")
print(f"   ln(x) = {np.log(x).round(4)}")
print(f"   log10(x) = {np.log10(x).round(4)}")

# 聚合函数
print("\n3. 聚合函数:")
arr = np.random.rand(3, 4)
print(f"   数组:\n{arr.round(4)}")
print(f"   sum: {arr.sum():.4f}")
print(f"   mean: {arr.mean():.4f}")
print(f"   std: {arr.std():.4f}")
print(f"   min: {arr.min():.4f}")
print(f"   max: {arr.max():.4f}")
print(f"   argmin: {arr.argmin()}")
print(f"   argmax: {arr.argmax()}")
print(f"   cumprod:\n{arr.cumprod().round(4).reshape(3, 4)}")

# 比较和逻辑函数
print("\n4. 比较和逻辑函数:")
a = np.array([1, 2, 3, 4, 5])
b = np.array([2, 2, 2, 2, 2])
print(f"   a = {a}, b = {b}")
print(f"   a > b: {a > b}")
print(f"   np.any(a > 3): {np.any(a > 3)}")
print(f"   np.all(a > 0): {np.all(a > 0)}")
print(f"   np.where(a > 2, a, 0): {np.where(a > 2, a, 0)}")

# 性能优化
print("\n=== 性能优化 ===")

# 1. 使用视图而非副本
print("1. 使用视图 (View) 而非副本 (Copy):")
arr = np.arange(10)
view = arr[::2]  # 这是一个视图
view[0] = 100
print(f"   原始数组: {arr}")
print(f"   修改视图后: {arr} (原始数组也改变了)")

arr2 = np.arange(10)
copy = arr2[::2].copy()  # 这是副本
copy[0] = 100
print(f"   原始数组: {arr2} (未改变)")

# 2. 使用in-place操作
print("\n2. In-place操作:")
arr = np.ones(1000000)
start = time.time()
arr = arr + 1
end = time.time()
normal_time = end - start

arr = np.ones(1000000)
start = time.time()
arr += 1
end = time.time()
inplace_time = end - start

print(f"   普通加法: {normal_time:.4f} 秒")
print(f"   In-place加法: {inplace_time:.4f} 秒")
print(f"   速度提升: {normal_time / inplace_time:.2f}x")

# 3. 内存使用
print("\n3. 内存使用:")
arr_int8 = np.zeros(1000000, dtype=np.int8)
arr_int32 = np.zeros(1000000, dtype=np.int32)
arr_float64 = np.zeros(1000000, dtype=np.float64)

print(f"   int8数组: {arr_int8.nbytes / 1024:.2f} KB")
print(f"   int32数组: {arr_int32.nbytes / 1024:.2f} KB")
print(f"   float64数组: {arr_float64.nbytes / 1024:.2f} KB")

# 4. 使用np.bincount
print("\n4. 使用np.bincount高效计数:")
arr = np.random.randint(0, 10, 1000000)
start = time.time()
counts_bincount = np.bincount(arr)
end = time.time()
bincount_time = end - start

start = time.time()
counts_loop = {i: (arr == i).sum() for i in range(10)}
end = time.time()
loop_time = end - start

print(f"   np.bincount: {bincount_time:.6f} 秒")
print(f"   循环计数: {loop_time:.6f} 秒")
print(f"   速度提升: {loop_time / bincount_time:.2f}x")

# 实际应用示例 - 矩阵运算
print("\n=== 实际应用示例 ===")

# 创建一个示例：图像卷积
print("图像卷积（简化版）:")
# 创建一个简单的"图像"
image = np.random.randn(100, 100)
# 创建一个简单的卷积核
kernel = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])  # Sobel算子

print(f"   图像形状: {image.shape}")
print(f"   卷积核:\n{kernel}")
print("   （实际使用scipy.ndimage.convolve进行完整卷积）")

# 矩阵运算性能对比
print("\n矩阵运算性能对比:")
size = 500
A = np.random.randn(size, size)
B = np.random.randn(size, size)

# Python循环方式（极慢，仅演示原理）
print(f"   矩阵形状: {A.shape} x {B.shape}")

# NumPy矩阵乘法
start = time.time()
C_numpy = A @ B
end = time.time()
numpy_time = end - start
print(f"   NumPy矩阵乘法: {numpy_time:.4f} 秒")

# 总结
print("\n=== NumPy高级数值计算学习总结 ===")
print("1. 广播机制 - 不同形状数组的运算")
print("2. 向量化操作 - 避免Python循环")
print("3. 高级索引 - 整数、布尔、花式索引")
print("4. 线性代数 - 矩阵运算、特征值、SVD")
print("5. 随机数生成 - 多种分布和采样方法")
print("6. 数组重塑和变换 - reshape、transpose、swapaxes")
print("7. 数组连接和拆分 - vstack、hstack、concatenate")
print("8. 通用函数 (ufunc) - 三角函数、指数、对数")
print("9. 聚合函数 - sum、mean、std等")
print("10. 性能优化 - 视图、in-place操作、内存优化")

print("\nNumPy高级数值计算学习完成！")
