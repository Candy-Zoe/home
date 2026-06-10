# Pandas数据可视化与统计学习
# 主要内容：内置绘图、统计函数、假设检验、窗口函数

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建示例数据 ===")
dates = pd.date_range('2020-01-01', periods=100, freq='D')
df = pd.DataFrame({
    'date': dates,
    'value1': np.random.randn(100).cumsum(),
    'value2': np.random.randn(100).cumsum(),
    'category': np.random.choice(['A', 'B', 'C'], 100)
})
df = df.set_index('date')

print(df.head())

print("\n=== 内置绘图 ===")
df['value1'].plot(title='时间序列图', figsize=(10, 5))
plt.xlabel('日期')
plt.ylabel('值')
plt.show()

print("\n=== 箱线图 ===")
df.boxplot(column='value1', by='category')
plt.title('箱线图')
plt.show()

print("\n=== 散点图矩阵 ===")
pd.plotting.scatter_matrix(df[['value1', 'value2']], figsize=(10, 5))
plt.show()

print("\n=== 滞后图 ===")
pd.plotting.lag_plot(df['value1'])
plt.title('滞后图')
plt.show()

print("\n=== 自相关图 ===")
pd.plotting.autocorrelation_plot(df['value1'])
plt.title('自相关图')
plt.show()

print("\n=== 统计函数 ===")
print(f"描述统计:\n{df.describe()}")

print(f"\n偏度: {df['value1'].skew()}")
print(f"峰度: {df['value1'].kurt()}")

print("\n=== 滚动统计 ===")
df['rolling_mean'] = df['value1'].rolling(window=10).mean()
df['rolling_std'] = df['value1'].rolling(window=10).std()

plt.figure(figsize=(12, 5))
plt.plot(df['value1'], label='原始值')
plt.plot(df['rolling_mean'], label='滚动均值')
plt.plot(df['rolling_std'], label='滚动标准差')
plt.legend()
plt.title('滚动统计')
plt.show()

print("\n=== 指数加权移动平均 ===")
df['ewma'] = df['value1'].ewm(span=10).mean()
plt.figure(figsize=(10, 5))
plt.plot(df['value1'], label='原始值')
plt.plot(df['ewma'], label='EWMA')
plt.legend()
plt.title('指数加权移动平均')
plt.show()

print("\n=== 扩展统计 ===")
df['expanding_mean'] = df['value1'].expanding().mean()
plt.figure(figsize=(10, 5))
plt.plot(df['value1'], label='原始值')
plt.plot(df['expanding_mean'], label='扩展均值')
plt.legend()
plt.title('扩展统计')
plt.show()

print("\n=== 排名函数 ===")
df['rank'] = df['value1'].rank()
print(f"排名:\n{df['rank'].head(10)}")

print("\n=== 百分位数排名 ===")
df['pct_rank'] = df['value1'].rank(pct=True)
print(f"百分位排名:\n{df['pct_rank'].head(10)}")

print("\n=== 变化率 ===")
df['pct_change'] = df['value1'].pct_change()
df['diff'] = df['value1'].diff()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(df['pct_change'])
plt.title('百分比变化')

plt.subplot(1, 2, 2)
plt.plot(df['diff'])
plt.title('差分')
plt.tight_layout()
plt.show()