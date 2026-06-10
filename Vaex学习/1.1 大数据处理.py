# Vaex大数据处理学习
# 主要内容：DataFrame操作、延迟计算、可视化

import vaex
import numpy as np

print("=== 创建Vaex DataFrame ===")
n = 1_000_000
df = vaex.from_arrays(
    x=np.random.randn(n),
    y=np.random.randn(n),
    z=np.random.rand(n)
)
print(f"DataFrame形状: {df.shape}")
print(f"DataFrame前5行:\n{df.head()}")

print("\n=== 基本操作 ===")
print(f"列名: {df.columns}")
print(f"描述统计:\n{df.describe()}")

print("\n=== 延迟计算 ===")
result = (df.x + df.y).mean()
print(f"延迟计算结果: {result}")

print("\n=== 过滤数据 ===")
filtered = df[(df.x > 0) & (df.y < 0)]
print(f"过滤后行数: {len(filtered)}")

print("\n=== 分组聚合 ===")
df['category'] = np.random.choice(['A', 'B', 'C'], n)
grouped = df.groupby('category', agg={'x_mean': vaex.agg.mean(df.x), 'count': vaex.agg.count()})
print(f"分组聚合结果:\n{grouped}")

print("\n=== 可视化 ===")
df.plot1d(df.x, figsize=(10, 5))
print("直方图已绘制")

df.plot2d(df.x, df.y, show=True)
print("2D分布图已绘制")

print("\n=== 内存效率 ===")
print(f"内存使用: {df.memory_usage().human_readable}")

print("\n=== 导出数据 ===")
df.export('test.hdf5')
print("数据已导出")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('test.hdf5'):
    os.remove('test.hdf5')
    print("已删除测试文件")