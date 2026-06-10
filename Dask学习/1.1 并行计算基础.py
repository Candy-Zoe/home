# Dask并行计算基础学习
# 主要内容：Dask数组、延迟计算、并行化

import dask.array as da
import numpy as np

print("=== 创建Dask数组 ===")
x = da.ones((10000, 10000), chunks=(1000, 1000))
print(f"Dask数组形状: {x.shape}")
print(f"Dask数组块大小: {x.chunks}")

print("\n=== 执行计算 ===")
y = x + x.T
z = y.mean()
result = z.compute()
print(f"计算结果: {result}")

print("\n=== Dask DataFrame ===")
import dask.dataframe as dd

df = dd.from_pandas(
    pd.DataFrame({'a': np.random.rand(10000), 'b': np.random.rand(10000)}),
    chunksize=1000
)
print(f"Dask DataFrame分区数: {df.npartitions}")

result = df.groupby('a').sum().compute()
print(f"分组聚合结果:\n{result.head()}")

print("\n=== 延迟计算 ===")
from dask import delayed

@delayed
def add(a, b):
    return a + b

@delayed
def multiply(a, b):
    return a * b

x = add(1, 2)
y = multiply(x, 3)
result = y.compute()
print(f"延迟计算结果: {result}")

print("\n=== 并行化循环 ===")
def process(i):
    return i * 2

results = [delayed(process)(i) for i in range(10)]
total = delayed(sum)(results)
print(f"并行循环结果: {total.compute()}")

import pandas as pd