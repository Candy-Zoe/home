# Pandas时间序列处理学习
# 主要内容：时间索引、日期范围、滚动窗口、重采样

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建时间序列 ===")
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100), index=dates)
print(f"时间序列前5行:\n{ts.head()}")

print("\n=== 时间索引 ===")
print(f"2023年1月数据:\n{ts['2023-01'].head()}")
print(f"2023年1月到2月数据:\n{ts['2023-01':'2023-02'].head()}")

print("\n=== 滚动窗口 ===")
rolling_mean = ts.rolling(window=7).mean()
plt.figure(figsize=(10, 4))
plt.plot(ts, label='原始数据')
plt.plot(rolling_mean, label='7日移动平均')
plt.legend()
plt.title('滚动窗口示例')
plt.show()

print("\n=== 指数加权移动平均 ===")
ewm = ts.ewm(span=7).mean()
plt.figure(figsize=(10, 4))
plt.plot(ts, label='原始数据')
plt.plot(ewm, label='EWMA')
plt.legend()
plt.title('指数加权移动平均')
plt.show()

print("\n=== 重采样 ===")
monthly = ts.resample('M').mean()
print(f"月度重采样:\n{monthly}")

weekly = ts.resample('W').sum()
print(f"\n周度重采样:\n{weekly}")

print("\n=== 时间差 ===")
diff = ts.diff()
print(f"差分后前5行:\n{diff.head()}")

print("\n=== 时间偏移 ===")
shifted = ts.shift(1)
print(f"滞后1天:\n{shifted.head()}")