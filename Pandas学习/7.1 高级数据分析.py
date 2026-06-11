# Pandas高级数据分析学习
# 主要内容：高级索引、透视表、时间序列、分组聚合、数据合并、缺失值处理

# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建示例数据
print("=== 创建示例数据 ===")

# 创建销售数据
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
products = ['产品A', '产品B', '产品C', '产品D']
regions = ['华东', '华南', '华北', '西南', '西北']

sales_data = pd.DataFrame({
    '日期': dates,
    '产品': np.random.choice(products, size=100),
    '区域': np.random.choice(regions, size=100),
    '销售额': np.random.randint(1000, 10000, size=100),
    '数量': np.random.randint(10, 100, size=100),
    '销售员': np.random.choice(['张三', '李四', '王五', '赵六'], size=100)
})

# 添加缺失值
sales_data.loc[np.random.choice(100, 10), '销售额'] = np.nan
sales_data.loc[np.random.choice(100, 5), '数量'] = np.nan

print(f"数据形状: {sales_data.shape}")
print(f"列名: {list(sales_data.columns)}")
print(f"\n前5行数据:")
print(sales_data.head())

# 基本统计
print(f"\n基本统计:")
print(f"总记录数: {len(sales_data)}")
print(f"唯一产品数: {sales_data['产品'].nunique()}")
print(f"唯一区域数: {sales_data['区域'].nunique()}")
print(f"唯一销售员数: {sales_data['销售员'].nunique()}")

# 高级索引
print("\n=== 高级索引 ===")

# 设置日期为索引
sales_indexed = sales_data.set_index('日期')
print("设置日期为索引:")
print(sales_indexed.head())

# 多级索引
sales_multiindex = sales_data.set_index(['日期', '区域', '产品'])
print("\n多级索引示例:")
print(sales_multiindex.head())

# 使用.loc选择特定日期范围
print("\n使用.loc选择日期范围:")
jan_sales = sales_indexed.loc['2023-01-15':'2023-01-31']
print(f"1月下半月销售额: {jan_sales['销售额'].sum():,.0f}")

# 使用布尔索引
print("\n布尔索引示例:")
high_sales = sales_data[sales_data['销售额'] > 7000]
print(f"销售额>7000的记录数: {len(high_sales)}")

# 组合条件
high_sales_east = sales_data[(sales_data['销售额'] > 7000) & (sales_data['区域'] == '华东')]
print(f"华东地区销售额>7000的记录数: {len(high_sales_east)}")

# 分组聚合
print("\n=== 分组聚合 ===")

# 按产品分组
print("\n按产品分组统计:")
product_group = sales_data.groupby('产品').agg({
    '销售额': ['sum', 'mean', 'std', 'count'],
    '数量': ['sum', 'mean']
})
print(product_group.round(2))

# 按区域分组
print("\n按区域分组统计:")
region_group = sales_data.groupby('区域').agg({
    '销售额': ['sum', 'mean', 'max', 'min'],
    '数量': 'sum'
})
print(region_group.round(2))

# 多重分组
print("\n按区域和产品双重分组:")
multi_group = sales_data.groupby(['区域', '产品']).agg({
    '销售额': ['sum', 'mean'],
    '数量': 'sum'
})
print(multi_group.round(2).head(10))

# 自定义聚合函数
print("\n自定义聚合函数:")
def range_func(x):
    """计算范围"""
    return x.max() - x.min()

custom_agg = sales_data.groupby('产品').agg({
    '销售额': ['sum', 'mean', range_func],
    '数量': ['sum', range_func]
})
print(custom_agg.round(2))

# 使用apply进行自定义操作
print("\n使用apply进行自定义操作:")
def z_score(x):
    """计算Z分数"""
    return (x - x.mean()) / x.std()

sales_data['销售额_Z分数'] = sales_data.groupby('区域')['销售额'].transform(z_score)
print("销售额Z分数计算完成")

# 透视表
print("\n=== 透视表 ===")

# 基本透视表
print("\n基本透视表（产品x区域，销售额均值）:")
pivot_table = pd.pivot_table(
    sales_data,
    values='销售额',
    index='产品',
    columns='区域',
    aggfunc='mean'
)
print(pivot_table.round(2))

# 多值透视表
print("\n多值透视表（销售额和数量）:")
pivot_multi = pd.pivot_table(
    sales_data,
    values=['销售额', '数量'],
    index='产品',
    columns='区域',
    aggfunc='sum'
)
print(pivot_multi.round(2))

# 透视表与边际合计
print("\n带边际合计的透视表:")
pivot_margins = pd.pivot_table(
    sales_data,
    values='销售额',
    index=['产品', '区域'],
    columns='销售员',
    aggfunc='sum',
    margins=True,
    margins_name='合计'
)
print(pivot_margins.round(2).head())

# 交叉表
print("\n=== 交叉表 ===")

# 基本交叉表
print("基本交叉表（产品x区域计数）:")
cross_table = pd.crosstab(sales_data['产品'], sales_data['区域'])
print(cross_table)

# 带百分比的交叉表
print("\n带百分比的交叉表:")
cross_percent = pd.crosstab(sales_data['产品'], sales_data['区域'], normalize='index') * 100
print(cross_percent.round(2))

# 数据合并
print("\n=== 数据合并 ===")

# 创建额外数据
product_info = pd.DataFrame({
    '产品': products,
    '单价': [100, 150, 200, 250],
    '库存': [500, 300, 200, 100]
})

region_info = pd.DataFrame({
    '区域': regions,
    '客户数': [1000, 800, 600, 500, 300],
    '市场份额': [0.30, 0.25, 0.20, 0.15, 0.10]
})

print("产品信息表:")
print(product_info)
print("\n区域信息表:")
print(region_info)

# Merge操作
print("\nMerge操作（合并产品信息）:")
merged_data = pd.merge(sales_data, product_info, on='产品', how='left')
print(f"合并后数据形状: {merged_data.shape}")
print(merged_data[['日期', '产品', '销售额', '数量', '单价', '库存']].head())

# 多次Merge
print("\n多次Merge（同时合并产品和区域信息）:")
merged_data2 = pd.merge(merged_data, region_info, on='区域', how='left')
print(f"合并后数据形状: {merged_data2.shape}")
print(merged_data2[['日期', '产品', '区域', '销售额', '数量', '单价', '库存', '客户数']].head())

# Join操作
print("\nJoin操作:")
sales_join = sales_data.set_index('产品').join(product_info.set_index('产品'))
print(f"Join后数据形状: {sales_join.shape}")

# Concat操作
print("\nConcat操作（拼接多个数据集）:")
df1 = sales_data[sales_data['产品'] == '产品A']
df2 = sales_data[sales_data['产品'] == '产品B']
concat_data = pd.concat([df1, df2], axis=0)
print(f"拼接后数据形状: {concat_data.shape}")

# 缺失值处理
print("\n=== 缺失值处理 ===")

# 查看缺失值
print("缺失值统计:")
print(sales_data.isnull().sum())
print(f"\n缺失值比例:")
print((sales_data.isnull().sum() / len(sales_data) * 100).round(2))

# 方法1: 删除缺失值
print("\n方法1: 删除包含缺失值的行")
sales_dropped = sales_data.dropna()
print(f"删除前: {len(sales_data)} 行")
print(f"删除后: {len(sales_dropped)} 行")
print(f"删除了 {len(sales_data) - len(sales_dropped)} 行")

# 方法2: 均值填充
print("\n方法2: 均值填充")
sales_mean_fill = sales_data.copy()
sales_mean_fill['销售额'] = sales_mean_fill['销售额'].fillna(sales_mean_fill['销售额'].mean())
sales_mean_fill['数量'] = sales_mean_fill['数量'].fillna(sales_mean_fill['数量'].mean())
print("均值填充完成")
print(f"销售额均值: {sales_data['销售额'].mean():.2f}")

# 方法3: 中位数填充
print("\n方法3: 中位数填充")
sales_median_fill = sales_data.copy()
sales_median_fill['销售额'] = sales_median_fill['销售额'].fillna(sales_median_fill['销售额'].median())
sales_median_fill['数量'] = sales_median_fill['数量'].fillna(sales_median_fill['数量'].median())
print("中位数填充完成")

# 方法4: 前向填充
print("\n方法4: 前向填充 (ffill)")
sales_ffill = sales_data.copy()
sales_ffill['销售额'] = sales_ffill['销售额'].ffill()
sales_ffill['数量'] = sales_ffill['数量'].ffill()
print("前向填充完成")

# 方法5: 后向填充
print("\n方法5: 后向填充 (bfill)")
sales_bfill = sales_data.copy()
sales_bfill['销售额'] = sales_bfill['销售额'].bfill()
sales_bfill['数量'] = sales_bfill['数量'].bfill()
print("后向填充完成")

# 方法6: 分组中位数填充
print("\n方法6: 按产品分组的中位数填充")
sales_group_fill = sales_data.copy()
for product in products:
    mask = (sales_group_fill['产品'] == product)
    product_median_sales = sales_group_fill.loc[mask, '销售额'].median()
    product_median_qty = sales_group_fill.loc[mask, '数量'].median()
    sales_group_fill.loc[mask, '销售额'] = sales_group_fill.loc[mask, '销售额'].fillna(product_median_sales)
    sales_group_fill.loc[mask, '数量'] = sales_group_fill.loc[mask, '数量'].fillna(product_median_qty)
print("分组中位数填充完成")

# 方法7: 插值法
print("\n方法7: 线性插值")
sales_interpolate = sales_data.copy()
sales_interpolate['销售额'] = sales_interpolate['销售额'].interpolate(method='linear')
sales_interpolate['数量'] = sales_interpolate['数量'].interpolate(method='linear')
print("线性插值完成")

# 不同填充方法的效果对比
print("\n=== 不同填充方法效果对比 ===")
print(f"原始销售额均值: {sales_data['销售额'].mean():.2f}")
print(f"均值填充后: {sales_mean_fill['销售额'].mean():.2f}")
print(f"中位数填充后: {sales_median_fill['销售额'].mean():.2f}")
print(f"前向填充后: {sales_ffill['销售额'].mean():.2f}")
print(f"分组中位数填充后: {sales_group_fill['销售额'].mean():.2f}")

# 时间序列分析
print("\n=== 时间序列分析 ===")

# 设置日期索引
sales_ts = sales_data.set_index('日期')
sales_ts['销售额'] = sales_ts['销售额'].fillna(sales_ts['销售额'].mean())
sales_ts['数量'] = sales_ts['数量'].fillna(sales_ts['数量'].mean())

# 按时间重采样
print("按周重采样:")
weekly_sales = sales_ts.resample('W').agg({
    '销售额': ['sum', 'mean'],
    '数量': 'sum'
})
print(weekly_sales.head().round(2))

# 按月重采样
print("\n按月重采样:")
monthly_sales = sales_ts.resample('M').agg({
    '销售额': 'sum',
    '数量': 'sum'
})
print(monthly_sales.round(2))

# 移动平均
print("\n移动平均 (7天窗口):")
sales_ts['销售额_MA7'] = sales_ts['销售额'].rolling(window=7).mean()
sales_ts['销售额_MA14'] = sales_ts['销售额'].rolling(window=14).mean()
print(sales_ts[['销售额', '销售额_MA7', '销售额_MA14']].tail(10).round(2))

# 指数加权移动平均
print("\n指数加权移动平均 (EWMA):")
sales_ts['销售额_EWMA7'] = sales_ts['销售额'].ewm(span=7).mean()
sales_ts['销售额_EWMA14'] = sales_ts['销售额'].ewm(span=14).mean()
print(sales_ts[['销售额', '销售额_EWMA7', '销售额_EWMA14']].tail(10).round(2))

# 日期特征提取
print("\n日期特征提取:")
sales_ts['年份'] = sales_ts.index.year
sales_ts['月份'] = sales_ts.index.month
sales_ts['星期'] = sales_ts.index.dayofweek
sales_ts['日'] = sales_ts.index.day
sales_ts['是否周末'] = sales_ts.index.dayofweek >= 5
print(sales_ts[['销售额', '年份', '月份', '星期', '是否周末']].head())

# 可视化时间序列
print("\n=== 时间序列可视化 ===")

# 销售额时间序列
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

axes[0].plot(sales_ts['销售额'], label='原始销售额')
axes[0].set_title('销售额时间序列')
axes[0].set_xlabel('日期')
axes[0].set_ylabel('销售额')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(sales_ts['销售额_MA7'], label='7天移动平均', color='orange')
axes[1].plot(sales_ts['销售额_MA14'], label='14天移动平均', color='green')
axes[1].set_title('移动平均')
axes[1].set_xlabel('日期')
axes[1].set_ylabel('销售额')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(sales_ts['销售额_EWMA7'], label='7天EWMA', color='purple')
axes[2].plot(sales_ts['销售额_EWMA14'], label='14天EWMA', color='red')
axes[2].set_title('指数加权移动平均')
axes[2].set_xlabel('日期')
axes[2].set_ylabel('销售额')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 分组时间序列
print("\n=== 分组时间序列分析 ===")

# 按区域分组的时间序列
region_time = pd.pivot_table(
    sales_data,
    values='销售额',
    index='日期',
    columns='区域',
    aggfunc='sum'
)
region_time = region_time.fillna(0)
region_time = region_time.resample('W').sum()

# 可视化
fig, ax = plt.subplots(figsize=(12, 6))
for region in region_time.columns:
    ax.plot(region_time.index, region_time[region], label=region)

ax.set_title('各区域周销售额趋势')
ax.set_xlabel('日期')
ax.set_ylabel('周销售额')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 高级数据操作
print("\n=== 高级数据操作 ===")

# 计算增长率
print("计算周增长率:")
weekly_growth = weekly_sales['销售额']['sum'].pct_change() * 100
print(weekly_growth.round(2).head())

# 计算累计和
print("\n计算累计销售额:")
sales_ts['累计销售额'] = sales_ts['销售额'].cumsum()
sales_ts['累计数量'] = sales_ts['数量'].cumsum()
print(sales_ts[['销售额', '累计销售额', '数量', '累计数量']].tail())

# 窗口函数
print("\n窗口函数 - 滚动统计:")
rolling_stats = sales_ts['销售额'].rolling(window=7).agg(['mean', 'std', 'min', 'max'])
print(rolling_stats.tail())

# 分位数
print("\n分位数分析:")
print(f"25%分位数: {sales_data['销售额'].quantile(0.25):.2f}")
print(f"50%分位数 (中位数): {sales_data['销售额'].quantile(0.50):.2f}")
print(f"75%分位数: {sales_data['销售额'].quantile(0.75):.2f}")
print(f"90%分位数: {sales_data['销售额'].quantile(0.90):.2f}")

# 排名
print("\n排名分析:")
sales_data['销售额排名'] = sales_data['销售额'].rank(ascending=False)
top_5_sales = sales_data.nlargest(5, '销售额')
print("销售额前5名:")
print(top_5_sales[['日期', '产品', '区域', '销售额', '销售额排名']])

# 数据透视和重塑
print("\n=== 数据透视和重塑 ===")

# melt操作 - 宽表转长表
wide_data = pd.pivot_table(
    sales_data,
    values='销售额',
    index='日期',
    columns='产品',
    aggfunc='sum'
).fillna(0)

print("宽表格式:")
print(wide_data.head())

# stack/unstack
print("\nStack/Unstack操作:")
stacked = wide_data.stack()
print(f"Stack后形状: {stacked.shape}")
print(stacked.head())

unstacked = stacked.unstack()
print(f"\nUnstack后形状: {unstacked.shape}")

# 性能优化技巧
print("\n=== 性能优化技巧 ===")

# 1. 使用更高效的数据类型
print("1. 优化数据类型:")
sales_optimized = sales_data.copy()
print(f"优化前内存使用: {sales_optimized.memory_usage(deep=True).sum() / 1024 / 1024:.4f} MB")

# 转换为category类型
sales_optimized['产品'] = sales_optimized['产品'].astype('category')
sales_optimized['区域'] = sales_optimized['区域'].astype('category')
sales_optimized['销售员'] = sales_optimized['销售员'].astype('category')

print(f"优化后内存使用: {sales_optimized.memory_usage(deep=True).sum() / 1024 / 1024:.4f} MB")

# 2. 使用eval进行高效计算
print("\n2. 使用eval进行高效计算:")
sales_optimized['均价'] = sales_optimized.eval('销售额 / 数量')
print("eval计算完成")

# 3. 使用query进行高效筛选
print("\n3. 使用query进行高效筛选:")
high_sales_query = sales_optimized.query('销售额 > 7000 and 区域 == "华东"')
print(f"query筛选结果: {len(high_sales_query)} 条记录")

# 4. 分块处理大数据
print("\n4. 分块处理 (适用于超大文件):")
print("提示: 对于大文件，可以使用chunksize参数逐块读取")

# 高级可视化
print("\n=== 高级可视化 ===")

# 分组柱状图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 产品销售对比
product_sales = sales_data.groupby('产品')['销售额'].sum()
axes[0, 0].bar(product_sales.index, product_sales.values)
axes[0, 0].set_title('各产品总销售额')
axes[0, 0].set_xlabel('产品')
axes[0, 0].set_ylabel('销售额')
axes[0, 0].tick_params(axis='x', rotation=45)

# 区域销售对比
region_sales = sales_data.groupby('区域')['销售额'].sum()
axes[0, 1].bar(region_sales.index, region_sales.values)
axes[0, 1].set_title('各区域总销售额')
axes[0, 1].set_xlabel('区域')
axes[0, 1].set_ylabel('销售额')
axes[0, 1].tick_params(axis='x', rotation=45)

# 销售分布
axes[1, 0].hist(sales_data['销售额'].dropna(), bins=30, edgecolor='black')
axes[1, 0].set_title('销售额分布')
axes[1, 0].set_xlabel('销售额')
axes[1, 0].set_ylabel('频数')

# 箱线图
data_to_plot = [sales_data[sales_data['产品'] == p]['销售额'].dropna() for p in products]
axes[1, 1].boxplot(data_to_plot)
axes[1, 1].set_xticklabels(products)
axes[1, 1].set_title('各产品销售额箱线图')
axes[1, 1].set_ylabel('销售额')

plt.tight_layout()
plt.show()

# 总结
print("\n=== Pandas高级数据分析学习总结 ===")
print("1. 高级索引技巧")
print("2. 分组聚合操作")
print("3. 透视表和交叉表")
print("4. 多种数据合并方法")
print("5. 缺失值处理策略")
print("6. 时间序列分析")
print("7. 数据透视和重塑")
print("8. 性能优化技巧")
print("9. 高级可视化")
print("10. 实用数据分析技巧")

print("\nPandas高级数据分析学习完成！")
