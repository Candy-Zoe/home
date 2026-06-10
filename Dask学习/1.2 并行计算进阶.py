# Dask并行计算进阶学习
# 主要内容：DataFrame进阶、机器学习集成、性能调优、分布式计算

import dask
import dask.array as da
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== 创建Dask集群 ===")
cluster = LocalCluster(n_workers=4, threads_per_worker=2)
client = Client(cluster)
print(f"集群状态: {client}")

print("\n=== Dask DataFrame ===")
df = pd.DataFrame({
    'x': np.random.randn(10000),
    'y': np.random.randn(10000),
    'group': np.random.choice(['A', 'B', 'C'], 10000)
})

ddf = dd.from_pandas(df, npartitions=10)
print(f"分区数: {ddf.npartitions}")

print("\n=== DataFrame操作 ===")
result = ddf.groupby('group').mean().compute()
print(f"分组均值:\n{result}")

print("\n=== 数据持久化 ===")
ddf_persisted = ddf.persist()
print("数据已持久化到集群")

print("\n=== Dask Array ===")
x = da.random.random((10000, 10000), chunks=(1000, 1000))
y = x + x.T
print(f"数组形状: {x.shape}")
print(f"分块形状: {x.chunks}")

print("\n=== 数组计算 ===")
result = y.mean().compute()
print(f"均值: {result:.4f}")

print("\n=== 机器学习集成 ===")
from dask_ml.linear_model import LinearRegression
from dask_ml.preprocessing import StandardScaler
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=10000, n_features=20, random_state=42)
X_dask = da.from_array(X, chunks=(1000, 20))
y_dask = da.from_array(y, chunks=(1000,))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dask)

model = LinearRegression()
model.fit(X_scaled, y_dask)
print("Dask机器学习模型训练完成")

print("\n=== 性能分析 ===")
from dask.diagnostics import Profiler, ResourceProfiler, CacheProfiler

with Profiler() as prof, ResourceProfiler() as res:
    result = ddf.groupby('group').mean().compute()

print(f"最耗时的操作:")
for key, value in prof.visualize().data.items()[:5]:
    print(f"  {key}: {value}")

print("\n=== 数据重分区 ===")
ddf_repartitioned = ddf.repartition(npartitions=5)
print(f"重分区后分区数: {ddf_repartitioned.npartitions}")

print("\n=== 时间序列处理 ===")
dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
ts_df = pd.DataFrame({
    'value': np.random.randn(1000),
    'date': dates
})
ts_df['date'] = pd.to_datetime(ts_df['date'])
ts_df = ts_df.set_index('date')

ddf_ts = dd.from_pandas(ts_df, npartitions=10)

daily = ddf_ts.resample('1D').mean()
result = daily.compute()
print(f"日均值:\n{result.head()}")

print("\n=== 图优化 ===")
x = da.random.random((1000, 1000), chunks=(500, 500))
y = x + 1
z = y * 2
w = z.mean()

print("计算图:")
print(dask.base.visualize(w))

print("\n=== 延迟计算 ===")
@dask.delayed
def slow_function(x):
    import time
    time.sleep(0.1)
    return x * 2

results = [slow_function(i) for i in range(10)]
total = dask.delayed(sum)(results)
result = total.compute()
print(f"延迟计算结果: {result}")

print("\n=== 关闭集群 ===")
client.close()
cluster.close()
print("集群已关闭")