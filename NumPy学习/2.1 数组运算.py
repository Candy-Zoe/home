# NumPy数组运算学习
# 主要内容：元素级运算、数学函数、比较运算、矩阵运算

# 导入NumPy库
import numpy as np

# 基本算术运算
print("=== 基本算术运算 ===")

# 创建测试数组
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

print(f"数组a: {a}")
print(f"数组b: {b}")

# 加法
print(f"加法 a + b: {a + b}")
print(f"加法 add(a, b): {np.add(a, b)}")

# 减法
print(f"减法 a - b: {a - b}")
print(f"减法 subtract(a, b): {np.subtract(a, b)}")

# 乘法（元素级乘法）
print(f"乘法 a * b: {a * b}")
print(f"乘法 multiply(a, b): {np.multiply(a, b)}")

# 除法
print(f"除法 b / a: {b / a}")
print(f"除法 divide(b, a): {np.divide(b, a)}")

# 整除
print(f"整除 b // a: {b // a}")

# 取模（余数）
print(f"取模 b % a: {b % a}")

# 幂运算
print(f"幂运算 a ** 2: {a ** 2}")
print(f"幂运算 power(a, 2): {np.power(a, 2)}")

# 标量运算
print("\n=== 标量运算 ===")

print(f"a + 10: {a + 10}")
print(f"a - 5: {a - 5}")
print(f"a * 2: {a * 2}")
print(f"a / 2: {a / 2}")

# 数学函数
print("\n=== 数学函数 ===")

# 三角函数
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(f"角度数组: {angles}")
print(f"正弦: {np.sin(angles)}")
print(f"余弦: {np.cos(angles)}")
print(f"正切: {np.tan(angles)}")

# 反三角函数
print(f"反正弦: {np.arcsin([0, 0.5, 1])}")

# 指数和对数
x = np.array([1, 2, 3])
print(f"\n数组x: {x}")
print(f"指数 exp: {np.exp(x)}")
print(f"自然对数 log: {np.log(x)}")
print(f"以2为底的对数 log2: {np.log2(x)}")
print(f"以10为底的对数 log10: {np.log10(x)}")

# 其他数学函数
print(f"平方根 sqrt: {np.sqrt(x)}")
print(f"绝对值 abs: {np.abs([-1, -2, 3])}")
print(f"符号函数 sign: {np.sign([-3, -1, 0, 1, 3])}")

# 聚合函数
print("\n=== 聚合函数 ===")

arr = np.arange(1, 10)
print(f"数组: {arr}")

print(f"求和 sum: {np.sum(arr)}")
print(f"均值 mean: {np.mean(arr)}")
print(f"标准差 std: {np.std(arr)}")
print(f"方差 var: {np.var(arr)}")
print(f"最小值 min: {np.min(arr)}")
print(f"最大值 max: {np.max(arr)}")
print(f"乘积 prod: {np.prod(arr)}")

# 累积函数
print("\n=== 累积函数 ===")

print(f"累积和 cumsum: {np.cumsum(arr)}")
print(f"累积积 cumprod: {np.cumprod(arr)}")

# 二维数组的聚合
print("\n=== 二维数组聚合 ===")

arr2d = np.arange(1, 10).reshape(3, 3)
print(f"二维数组:\n{arr2d}")

# 按行聚合（axis=1）
print(f"每行求和: {np.sum(arr2d, axis=1)}")
print(f"每行均值: {np.mean(arr2d, axis=1)}")

# 按列聚合（axis=0）
print(f"每列求和: {np.sum(arr2d, axis=0)}")
print(f"每列均值: {np.mean(arr2d, axis=0)}")

# 比较运算
print("\n=== 比较运算 ===")

a = np.array([1, 2, 3, 4, 5])
b = np.array([5, 4, 3, 2, 1])

print(f"数组a: {a}")
print(f"数组b: {b}")

print(f"相等: {a == b}")
print(f"不等: {a != b}")
print(f"大于: {a > b}")
print(f"小于: {b < a}")
print(f"大于等于: {a >= 3}")
print(f"小于等于: {b <= 3}")

# 逻辑运算
print("\n=== 逻辑运算 ===")

a = np.array([True, False, True, False])
b = np.array([True, True, False, False])

print(f"数组a: {a}")
print(f"数组b: {b}")

print(f"逻辑与: {np.logical_and(a, b)}")
print(f"逻辑或: {np.logical_or(a, b)}")
print(f"逻辑非: {np.logical_not(a)}")
print(f"逻辑异或: {np.logical_xor(a, b)}")

# 矩阵运算
print("\n=== 矩阵运算 ===")

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")

# 矩阵乘法（使用@运算符或np.matmul）
print(f"矩阵乘法 A @ B:\n{A @ B}")
print(f"矩阵乘法 np.matmul(A, B):\n{np.matmul(A, B)}")

# 矩阵转置
print(f"矩阵A的转置:\n{A.T}")

# 矩阵求逆
print(f"矩阵A的逆:\n{np.linalg.inv(A)}")

# 矩阵行列式
print(f"矩阵A的行列式: {np.linalg.det(A)}")

# 矩阵特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 向量点积和叉积
print("\n=== 向量运算 ===")

u = np.array([1, 2, 3])
v = np.array([4, 5, 6])

print(f"向量u: {u}")
print(f"向量v: {v}")

# 点积
print(f"点积: {np.dot(u, v)}")
print(f"点积（使用@）: {u @ v}")

#叉积（仅适用于3D向量）
print(f"叉积: {np.cross(u, v)}")

# 范数
print(f"向量u的L2范数: {np.linalg.norm(u)}")

# 广播机制
print("\n=== 广播机制 ===")

a = np.array([[1, 2, 3], [4, 5, 6]])  # 形状(2, 3)
b = np.array([10, 20, 30])  # 形状(3,)

print(f"数组a:\n{a}")
print(f"数组b: {b}")
print(f"a + b:\n{a + b}")