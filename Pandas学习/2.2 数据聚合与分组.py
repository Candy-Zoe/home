# Pandas数据聚合与分组学习
# 主要内容：groupby分组、聚合函数、变换和过滤

# 导入Pandas库
import pandas as pd
import numpy as np

# 创建示例销售数据
print("=== 创建示例数据 ===")

data = {
    '日期': pd.date_range('2023-01-01', periods=20, freq='D'),
    '城市': ['北京', '上海', '北京', '上海', '北京', '上海', '北京', '上海',
            '北京', '上海', '北京', '上海', '北京', '上海', '北京', '上海',
            '北京', '上海', '北京', '上海'],
    '产品': ['手机', '电脑', '平板', '手机', '电脑', '平板', '手机', '电脑',
            '平板', '手机', '电脑', '平板', '手机', '电脑', '平板', '手机',
            '电脑', '平板', '手机', '电脑'],
    '销售额': np.random.randint(1000, 5000, 20),
    '数量': np.random.randint(10, 100, 20)
}

df = pd.DataFrame(data)

# 添加一些计算字段
df['单价'] = df['销售额'] / df['数量']

print(df.head(10))
print(f"\n数据形状: {df.shape}")

# 基本分组操作
print("\n=== 基本分组 ===")

# 按单列分组
grouped = df.groupby('城市')
print(f"分组类型: {type(grouped)}")
print(f"分组数量: {grouped.ngroups}")

# 查看分组
for name, group in grouped:
    print(f"\n{name}: {len(group)} 条记录")

# 按多列分组
print("\n=== 多列分组 ===")

grouped_multi = df.groupby(['城市', '产品'])
print(f"多列分组数量: {grouped_multi.ngroups}")

# 聚合操作
print("\n=== 聚合操作 ===")

# 使用agg进行多个聚合
agg_result = df.groupby('城市').agg({
    '销售额': ['sum', 'mean', 'max', 'min'],
    '数量': ['sum', 'mean'],
    '单价': 'mean'
})

print("按城市聚合的结果:")
print(agg_result)

# 使用agg命名聚合结果
named_agg = df.groupby('城市').agg(
    总销售额=('销售额', 'sum'),
    平均销售额=('销售额', 'mean'),
    最大销售额=('销售额', 'max'),
    总数量=('数量', 'sum')
)

print("\n命名聚合结果:")
print(named_agg)

# 使用transform进行数据变换
print("\n=== 数据变换 ===")

# 计算每个城市的销售额占总额比例
df['城市销售额占比'] = df.groupby('城市')['销售额'].transform(lambda x: x / x.sum() * 100)
print("添加城市销售额占比后:")
print(df[['城市', '产品', '销售额', '城市销售额占比']].head())

# 计算每个城市的销售额排名
df['城市内排名'] = df.groupby('城市')['销售额'].transform(lambda x: x.rank(ascending=False))
print("\n添加城市内排名后:")
print(df[['城市', '产品', '销售额', '城市内排名']].head(10))

# 使用apply进行复杂变换
print("\n=== apply自定义聚合 ===")

# 定义自定义聚合函数
def top_sales(series, n=2):
    """返回销售额最高的前n个值"""
    return series.nlargest(n)

result = df.groupby('城市').apply(lambda x: x.nlargest(2, '销售额'))
print("每个城市销售额最高的2条记录:")
print(result)

# 过滤操作
print("\n=== 分组过滤 ===")

# 过滤销售额总和大于平均值的城市
city_stats = df.groupby('城市').agg({
    '销售额': 'sum',
    '数量': 'sum'
})
print("城市统计:")
print(city_stats)

# 计算平均销售额
avg_sales = city_stats['销售额'].mean()
print(f"\n平均总销售额: {avg_sales:.2f}")

# 过滤
filtered_cities = city_stats[city_stats['销售额'] > avg_sales]
print(f"销售额高于平均值的城市:")
print(filtered_cities)

# 使用filter方法进行分组过滤
print("\n使用filter方法:")
def filter_func(group):
    return group['销售额'].sum() > avg_sales

filtered_df = df.groupby('城市').filter(filter_func)
print(f"过滤后的数据形状: {filtered_df.shape}")

# 透视表
print("\n=== 透视表 ===")

# 创建透视表
pivot = pd.pivot_table(
    df,
    values='销售额',
    index='城市',
    columns='产品',
    aggfunc='sum',
    fill_value=0
)

print("透视表（销售额）:")
print(pivot)

# 添加小计
pivot_with_margins = pd.pivot_table(
    df,
    values='销售额',
    index='城市',
    columns='产品',
    aggfunc='sum',
    fill_value=0,
    margins=True,
    margins_name='总计'
)

print("\n带小计的透视表:")
print(pivot_with_margins)

# 交叉表
print("\n=== 交叉表 ===")

# 创建交叉表
crosstab = pd.crosstab(
    df['城市'],
    df['产品'],
    margins=True
)

print("交叉表:")
print(crosstab)

# 添加值
crosstab_values = pd.crosstab(
    df['城市'],
    df['产品'],
    values=df['销售额'],
    aggfunc='sum',
    normalize='index'
)

print("\n归一化交叉表（按行）:")
print(crosstab_values)

# 时间序列分组
print("\n=== 时间序列分组 ===")

# 按月份分组
df['月份'] = df['日期'].dt.month
monthly_sales = df.groupby('月份')['销售额'].sum()
print("按月份销售总额:")
print(monthly_sales)

# 按星期分组
df['星期'] = df['日期'].dt.day_name()
weekly_sales = df.groupby('星期')['销售额'].sum()
print("\n按星期销售总额:")
print(weekly_sales)

# 综合示例
print("\n=== 综合示例 ===")

# 创建销售报表
report = df.groupby(['城市', '产品']).agg({
    '销售额': ['sum', 'mean', 'std'],
    '数量': ['sum', 'mean']
}).round(2)

report.columns = ['_'.join(col) for col in report.columns]
report = report.reset_index()
print("销售报表:")
print(report)

# 计算同比增长
print("\n=== 计算字段 ===")

# 计算利润率（假设成本为销售额的60%）
df['成本'] = df['销售额'] * 0.6
df['利润'] = df['销售额'] - df['成本']
df['利润率'] = (df['利润'] / df['销售额'] * 100).round(2)

print("添加利润信息后:")
print(df[['城市', '产品', '销售额', '成本', '利润', '利润率']].head(10))

# 按城市计算平均利润率
profit_report = df.groupby('城市').agg({
    '销售额': 'sum',
    '利润': 'sum'
})
profit_report['平均利润率'] = (profit_report['利润'] / profit_report['销售额'] * 100).round(2)
print("\n按城市利润报告:")
print(profit_report)