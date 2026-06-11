# Pandas时间序列处理学习
# 主要内容：时间索引、重采样、滑动窗口、时区处理

# 导入Pandas库
import pandas as pd
import numpy as np

# 创建时间序列数据
print("=== 创建时间序列数据 ===")

# 方法1：使用date_range创建日期范围
dates = pd.date_range('2023-01-01', periods=30, freq='D')
print(f"日期范围: {dates[:5]}")

# 创建时间序列DataFrame
df = pd.DataFrame({
    '日期': dates,
    '销售额': np.random.randint(1000, 5000, 30),
    '访问量': np.random.randint(100, 1000, 30)
})
df.set_index('日期', inplace=True)

print("\n时间序列数据:")
print(df.head(10))

# 方法2：使用DatetimeIndex
print("\n=== DatetimeIndex ===")

# 从字符串创建
dates_str = ['2023-01-01', '2023-01-02', '2023-01-03']
dt_index = pd.to_datetime(dates_str)
print(f"从字符串创建: {dt_index}")

# 从多种格式解析
mixed_dates = ['2023/01/01', '2023-01-02', '03/01/2023', 'Jan 04, 2023']
parsed = pd.to_datetime(mixed_dates)
print(f"解析混合格式: {parsed}")

# 创建不同频率的日期范围
print("\n不同频率的日期范围:")

# 每小时
hourly = pd.date_range('2023-01-01', periods=24, freq='H')
print(f"每小时: {len(hourly)} 个时间点")

# 每月
monthly = pd.date_range('2023-01-01', periods=12, freq='M')
print(f"每月: {monthly.tolist()}")

# 每工作日
business = pd.date_range('2023-01-01', periods=22, freq='B')
print(f"每工作日: {len(business)} 个工作日")

# 时间序列索引和切片
print("\n=== 时间序列索引和切片 ===")

# 创建更长时间序列
date_range = pd.date_range('2023-01-01', periods=365, freq='D')
ts_df = pd.DataFrame({
    '值': np.random.randn(365).cumsum()
}, index=date_range)

# 按日期选择
print(f"选择2023年1月的数据: {ts_df.loc['2023-01'].head()}")

# 按范围选择
print(f"\n选择2023年1月到2月的数据: {ts_df.loc['2023-01':'2023-02'].head()}")

# 按年份选择
print(f"\n选择2023年的数据: {len(ts_df.loc['2023'])} 天")

# 重采样
print("\n=== 重采样 ===")

# 创建日数据
daily_data = pd.DataFrame({
    '销售额': np.random.randint(1000, 5000, 90)
}, index=pd.date_range('2023-01-01', periods=90, freq='D'))

# 重采样为周数据
weekly = daily_data.resample('W').sum()
print(f"周销售额统计:\n{weekly.head()}")

# 重采样为月数据
monthly_sum = daily_data.resample('M').sum()
monthly_mean = daily_data.resample('M').mean()
monthly_std = daily_data.resample('M').std()

print(f"\n月销售额统计:")
stats_df = pd.DataFrame({
    '总和': monthly_sum['销售额'],
    '均值': monthly_mean['销售额'],
    '标准差': monthly_std['销售额']
})
print(stats_df)

# 向下采样（聚合）
print("\n向下采样示例:")
hourly_data = pd.DataFrame({
    '温度': np.random.randint(20, 35, 72)
}, index=pd.date_range('2023-01-01', periods=72, freq='H'))

# 按6小时聚合
six_hourly = hourly_data.resample('6H').mean()
print(six_hourly)

# 滑动窗口
print("\n=== 滑动窗口 ===")

# 创建时间序列
dates = pd.date_range('2023-01-01', periods=100, freq='D')
ts = pd.Series(np.random.randn(100).cumsum(), index=dates)

# 滑动窗口统计
window_sizes = [7, 14, 30]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 原始数据
axes[0, 0].plot(ts.index, ts.values, 'b-', linewidth=1)
axes[0, 0].set_title('原始数据')
axes[0, 0].grid(True, alpha=0.3)

# 不同窗口大小的移动平均
for i, window in enumerate(window_sizes):
    row, col = (i + 1) // 2, (i + 1) % 2
    rolling_mean = ts.rolling(window=window).mean()
    axes[row, col].plot(ts.index, ts.values, 'b-', alpha=0.3, label='原始')
    axes[row, col].plot(rolling_mean.index, rolling_mean.values, 'r-', 
                         linewidth=2, label=f'{window}日均线')
    axes[row, col].set_title(f'{window}日移动平均')
    axes[row, col].legend()
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 滑动窗口函数
print("\n滑动窗口统计函数:")

window = ts.rolling(window=7)
stats = pd.DataFrame({
    '均值': window.mean(),
    '标准差': window.std(),
    '最小值': window.min(),
    '最大值': window.max(),
    '中位数': window.median()
})
print(stats.head(14))

# 指数加权移动平均
print("\n=== 指数加权移动平均 ===")

# 简单移动平均 vs 指数加权移动平均
span = 7  # 相当于7日窗口
ewma = ts.ewm(span=span).mean()
sma = ts.rolling(window=span).mean()

print(f"简单移动平均（前7天）: {sma.iloc[6]:.4f}")
print(f"指数加权移动平均（span=7）: {ewma.iloc[6]:.4f}")

# 时区处理
print("\n=== 时区处理 ===")

# 创建带时区的时间序列
dates_tz = pd.date_range('2023-01-01', periods=10, freq='D', tz='Asia/Shanghai')
ts_tz = pd.Series(range(10), index=dates_tz)
print(f"带时区的序列:\n{ts_tz}")

# 转换时区
ts_utc = ts_tz.tz_convert('UTC')
print(f"\n转换为UTC:\n{ts_utc}")

# 本地化
dates_naive = pd.date_range('2023-01-01', periods=10, freq='D')
ts_naive = pd.Series(range(10), index=dates_naive)
ts_localized = ts_naive.tz_localize('Asia/Shanghai')
print(f"\n本地化到上海时区:\n{ts_localized}")

# 时间偏移
print("\n=== 时间偏移 ===")

# 创建带偏移的时间
dates = pd.date_range('2023-01-01', periods=10, freq='D')

# 添加偏移
offset_days = pd.DateOffset(days=5)
print(f"原日期 + 5天: {dates[0]} -> {dates[0] + offset_days}")

# 使用Timedelta
td = pd.Timedelta(days=5, hours=2)
print(f"原日期 + 5天2小时: {dates[0]} -> {dates[0] + td}")

# 时间差计算
print("\n时间差计算:")
date1 = pd.Timestamp('2023-01-01')
date2 = pd.Timestamp('2023-01-15')
diff = date2 - date1
print(f"{date2} - {date1} = {diff} ({diff.days}天)")

# 时期和频率转换
print("\n=== 时期和频率转换 ===")

# 创建月度时期
monthly_period = pd.period_range('2023-01', periods=12, freq='M')
period_df = pd.DataFrame({
    '月份': monthly_period,
    '销售额': np.random.randint(10000, 50000, 12)
})
print(f"月度时期:\n{period_df}")

# 时期转时间戳
period_df['时间戳'] = period_df['月份'].dt.to_timestamp()
print(f"\n转换为时间戳:\n{period_df}")

# 频率转换
period_df['开始日期'] = period_df['月份'].dt.start_time
period_df['结束日期'] = period_df['月份'].dt.end_time
print(f"\n开始和结束日期:\n{period_df[['月份', '开始日期', '结束日期']]}")

# 时间序列特征提取
print("\n=== 时间序列特征提取 ===")

dates = pd.date_range('2023-01-01', periods=365, freq='D')
ts = pd.DataFrame({
    '日期': dates,
    '值': np.random.randn(365).cumsum()
})
ts.set_index('日期', inplace=True)

# 提取时间特征
ts['年'] = ts.index.year
ts['月'] = ts.index.month
ts['日'] = ts.index.day
ts['星期'] = ts.index.dayofweek
ts['星期名'] = ts.index.day_name()
ts['季度'] = ts.index.quarter
ts['是否周末'] = ts.index.is_weekend()
ts['是否月初'] = ts.index.is_month_start
ts['是否月末'] = ts.index.is_month_end

print("时间序列特征:")
print(ts[['值', '年', '月', '星期', '星期名', '季度', '是否周末']].head(10))

# 分组聚合
print("\n=== 按时间特征分组聚合 ===")

# 按星期聚合
weekly_stats = ts.groupby('星期名')['值'].agg(['mean', 'std', 'count'])
print("按星期统计:")
print(weekly_stats)

# 按月聚合
monthly_stats = ts.groupby('月')['值'].sum()
print("\n按月统计:")
print(monthly_stats)

# 数据插值
print("\n=== 数据插值 ===")

# 创建有缺失值的时间序列
dates = pd.date_range('2023-01-01', periods=10, freq='D')
values = [1, 2, np.nan, 4, np.nan, 6, 7, np.nan, 9, 10]
ts_missing = pd.Series(values, index=dates)
print(f"有缺失值的时间序列:\n{ts_missing}")

# 线性插值
ts_interpolated = ts_missing.interpolate(method='linear')
print(f"\n线性插值后:\n{ts_interpolated}")

# 时间序列移动相关性
print("\n=== 移动相关性 ===")

# 创建两个相关的时间序列
np.random.seed(42)
ts1 = pd.Series(np.random.randn(100).cumsum())
ts2 = ts1 + np.random.randn(100) * 0.5

# 计算移动相关性
rolling_corr = ts1.rolling(window=20).corr(ts2)
print(f"20日移动相关性（前10个值）:\n{rolling_corr.head(10)}")

print("\n时间序列学习完成！")