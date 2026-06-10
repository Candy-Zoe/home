# NumPy随机数生成学习
# 主要内容：随机数生成器、概率分布、随机抽样

import numpy as np
import matplotlib.pyplot as plt

print("=== 设置随机种子 ===")
np.random.seed(42)

print("\n=== 均匀分布 ===")
uniform = np.random.rand(1000)
print(f"均匀分布样本: {uniform[:5]}")
plt.hist(uniform, bins=20)
plt.title('均匀分布')
plt.show()

print("\n=== 正态分布 ===")
normal = np.random.randn(1000)
print(f"正态分布样本: {normal[:5]}")
plt.hist(normal, bins=20)
plt.title('标准正态分布')
plt.show()

print("\n=== 指定均值和标准差的正态分布 ===")
mu, sigma = 5, 2
normal_custom = np.random.normal(mu, sigma, 1000)
print(f"均值: {np.mean(normal_custom):.2f}, 标准差: {np.std(normal_custom):.2f}")
plt.hist(normal_custom, bins=20)
plt.title(f'正态分布 (mu={mu}, sigma={sigma})')
plt.show()

print("\n=== 整数随机数 ===")
integers = np.random.randint(0, 10, size=10)
print(f"0-9的随机整数: {integers}")

print("\n=== 随机排列 ===")
arr = np.array([1, 2, 3, 4, 5])
np.random.shuffle(arr)
print(f"随机排列: {arr}")

print("\n=== 随机抽样 ===")
population = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
sample = np.random.choice(population, size=3, replace=False)
print(f"无放回抽样: {sample}")
sample_with_replace = np.random.choice(population, size=5, replace=True)
print(f"有放回抽样: {sample_with_replace}")