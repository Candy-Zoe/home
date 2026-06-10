# Vaex大数据处理进阶学习
# 主要内容：表达式系统、用户定义函数、性能优化、数据可视化

import vaex
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建示例数据 ===")
n = 1000000
df = vaex.from_dict({
    'x': np.random.randn(n),
    'y': np.random.randn(n),
    'z': np.random.randn(n),
    'category': np.random.choice(['A', 'B', 'C', 'D'], n),
    'value': np.random.randint(0, 100, n)
})

print(f"数据形状: {df.shape}")
print(f"列: {df.column_names}")

print("\n=== 表达式系统 ===")
df['x_squared'] = df.x ** 2
df['xy_sum'] = df.x + df.y
df['normalized'] = (df.x - df.x.mean()) / df.x.std()

print("表达式列已创建")

print("\n=== 高效过滤 ===")
df_filtered = df[df.x > 0]
print(f"过滤后形状: {df_filtered.shape}")

print("\n=== 聚合操作 ===")
result = df.groupby('category', agg={
    'count': vaex.agg.count(),
    'mean_x': vaex.agg.mean('x'),
    'std_x': vaex.agg.std('x'),
    'sum_value': vaex.agg.sum('value')
})
print(f"分组聚合结果:\n{result}")

print("\n=== 用户定义函数 ===")
@vaex.register_function()
def poly3(x):
    return x ** 3 + 2 * x ** 2 - x + 1

df.apply(poly3, arguments=[df.x], dtype=np.float64)
print("多项式函数已注册")

print("\n=== 滑动窗口 ===")
df['rolling_mean'] = df.x.rolling(10).mean()
print("滚动均值已计算")

print("\n=== 虚拟列 vs .materialize ===")
df['virtual'] = df.x + df.y
df.materialize('virtual', progress=True)
print("列已物化")

print("\n=== 数据选择 ===")
df_selection = df[df.category == 'A']
print(f"选择后形状: {df_selection.shape}")

print("\n=== 统计信息 ===")
print(df.describe())

print("\n=== 数据可视化 ===")
df.plot(df.x, df.y, what='count(*)')
plt.title('散点图')
plt.show()

print("\n=== 柱状图 ===")
df.plot1d(df.value, bins=50)
plt.title('直方图')
plt.show()

print("\n=== 热力图 ===")
df.plot(df.x, df.y, what=vaex.stat.count(), limits='minmax', colormap='viridis')
plt.title('热力图')
plt.show()

print("\n=== 导出数据 ===")
df_small = df[:1000]
df_small.export('sample.csv', progress=True)
print("数据已导出")

print("\n=== 读取数据 ===")
df_read = vaex.read('sample.csv')
print(f"读取数据形状: {df_read.shape}")

print("\n=== 性能基准测试 ===")
import time

start = time.time()
result = df.x.mean()
elapsed = time.time() - start
print(f"均值计算时间: {elapsed:.4f}s")

print("\n=== 数据采样 ===")
df_sample = df.sample(n=10000)
print(f"采样后形状: {df_sample.shape}")

df_frac_sample = df.sample(frac=0.01)
print(f"比例采样后形状: {df_frac_sample.shape}")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('sample.csv'):
    os.remove('sample.csv')
    print("已删除测试文件")