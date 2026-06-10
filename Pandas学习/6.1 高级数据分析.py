# Pandas高级数据分析学习
# 主要内容：多级索引、数据透视、时间序列分析、数据重塑

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建多级索引数据 ===")
arrays = [
    ['A', 'A', 'A', 'B', 'B', 'B'],
    ['X', 'Y', 'Z', 'X', 'Y', 'Z']
]
index = pd.MultiIndex.from_arrays(arrays, names=['Group', 'Type'])

df = pd.DataFrame({
    'Value1': np.random.randn(6),
    'Value2': np.random.randn(6)
}, index=index)

print(f"多级索引DataFrame:\n{df}")

print("\n=== 多级索引操作 ===")
print(f"选择Group=A:\n{df.loc['A']}")
print(f"\n选择Group=A, Type=X:\n{df.loc[('A', 'X')]}")

print("\n=== 数据透视 ===")
df_pivot = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=12),
    'Category': ['A', 'B', 'C'] * 4,
    'Value': np.random.randn(12) * 10 + 50
})

pivot_table = df_pivot.pivot_table(
    values='Value',
    index='Date',
    columns='Category',
    aggfunc='mean'
)
print(f"透视表:\n{pivot_table}")

print("\n=== 交叉表 ===")
df_cross = pd.DataFrame({
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F'] * 10,
    'Age': ['Young', 'Old', 'Young', 'Old', 'Young', 'Old'] * 10,
    'Buy': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No'] * 10
})

cross_tab = pd.crosstab(df_cross['Gender'], df_cross['Age'])
print(f"交叉表:\n{cross_tab}")

print("\n=== 数据重塑 ===")
df_melt = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Math': [90, 85, 88],
    'Physics': [85, 90, 82],
    'Chemistry': [88, 86, 90]
})

melted = df_melt.melt(id_vars='Name', var_name='Subject', value_name='Score')
print(f"融化后:\n{melted}")

pivoted = melted.pivot(index='Name', columns='Subject', values='Score')
print(f"\n重塑回宽格式:\n{pivoted}")

print("\n=== 时间序列分析 ===")
dates = pd.date_range('2024-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

print(f"时间序列前5个:\n{ts.head()}")

print("\n=== 时间序列重采样 ===")
monthly = ts.resample('M').mean()
print(f"月度均值:\n{monthly}")

print("\n=== 滚动窗口 ===")
rolling_mean = ts.rolling(window=7).mean()
rolling_std = ts.rolling(window=7).std()

plt.figure(figsize=(12, 5))
plt.plot(ts, label='原始')
plt.plot(rolling_mean, label='7日滚动均值')
plt.fill_between(ts.index, rolling_mean - rolling_std, rolling_mean + rolling_std, alpha=0.2)
plt.legend()
plt.title('时间序列分析')
plt.show()

print("\n=== 时间序列分解 ===")
from statsmodels.tsa.seasonal import seasonal_decompose

ts_decompose = pd.Series(
    10 + 0.1 * np.arange(100) + 5 * np.sin(np.arange(100) * 2 * np.pi / 12) + np.random.randn(100),
    index=dates
)

decomposition = seasonal_decompose(ts_decompose, period=12)

fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='原始')
decomposition.trend.plot(ax=axes[1], title='趋势')
decomposition.seasonal.plot(ax=axes[2], title='季节性')
decomposition.resid.plot(ax=axes[3], title='残差')
plt.tight_layout()
plt.show()

print("\n=== 数据合并进阶 ===")
df1 = pd.DataFrame({'key': ['A', 'B', 'C'], 'value1': [1, 2, 3]})
df2 = pd.DataFrame({'key': ['A', 'B', 'D'], 'value2': [4, 5, 6]})

inner_join = pd.merge(df1, df2, on='key', how='inner')
outer_join = pd.merge(df1, df2, on='key', how='outer')
left_join = pd.merge(df1, df2, on='key', how='left')

print(f"内连接:\n{inner_join}")
print(f"\n外连接:\n{outer_join}")
print(f"\n左连接:\n{left_join}")

print("\n=== 分组变换 ===")
df_group = pd.DataFrame({
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'],
    'Value': [1, 2, 3, 4, 5, 6]
})

df_group['Normalized'] = df_group.groupby('Category')['Value'].transform(
    lambda x: (x - x.mean()) / x.std()
)
print(f"分组标准化:\n{df_group}")