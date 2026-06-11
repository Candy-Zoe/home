# Pandas数据清洗学习
# 主要内容：处理缺失值、重复数据、异常值、数据类型转换

# 导入Pandas库
import pandas as pd
import numpy as np

# 创建包含各种问题的示例数据
print("=== 创建示例数据 ===")

data = {
    '姓名': ['张三', '李四', '王五', '赵六', '张三', '陈七', None, '吴八'],
    '年龄': [25, 30, 35, -5, 28, 42, 33, np.nan],
    '城市': ['北京', '上海', '北京', '深圳', '北京', None, '上海', '深圳'],
    '工资': [8000, 12000, 10000, 15000, 8000, 9000, 11000, 13500],
    '部门': ['技术', '销售', '技术', '人事', '技术', '销售', '人事', '销售'],
    '入职日期': ['2020-01-01', '2019-06-15', '2021-03-20', '2018-09-01', 
                '2020-01-01', '2022-01-10', '2019-12-05', '2021-07-15']
}

df = pd.DataFrame(data)
print("原始数据:")
print(df)
print(f"\n数据形状: {df.shape}")

# 处理缺失值
print("\n=== 处理缺失值 ===")

# 检测缺失值
print("各列缺失值数量:")
print(df.isnull().sum())

print("\n各列缺失值比例:")
print((df.isnull().sum() / len(df) * 100).round(2))

# 查看包含缺失值的行
print("\n包含缺失值的行:")
print(df[df.isnull().any(axis=1)])

# 删除缺失值
print("\n删除包含缺失值的行:")
df_dropped = df.dropna()
print(f"删除前: {len(df)} 行, 删除后: {len(df_dropped)} 行")

# 只删除指定列有缺失值的行
print("\n删除'年龄'列有缺失值的行:")
df_age_clean = df.dropna(subset=['年龄'])
print(df_age_clean)

# 填充缺失值
print("\n=== 填充缺失值 ===")

# 用均值填充数值列
df_filled = df.copy()
df_filled['年龄'].fillna(df['年龄'].mean(), inplace=True)
print("用均值填充'年龄'列:")
print(df_filled)

# 用众数填充分类列
df_filled['城市'].fillna(df['城市'].mode()[0], inplace=True)
print("\n用众数填充'城市'列:")
print(df_filled)

# 用前一个值填充（向前填充）
df_ffill = df.fillna(method='ffill')
print("\n向前填充:")
print(df_ffill)

# 用后一个值填充（向后填充）
df_bfill = df.fillna(method='bfill')
print("\n向后填充:")
print(df_bfill)

# 处理重复数据
print("\n=== 处理重复数据 ===")

# 检测重复行
print("重复行检测:")
print(df.duplicated())

# 显示重复行
print("\n重复行内容:")
print(df[df.duplicated(keep=False)])

# 删除重复行（保留第一个）
df_no_dup = df.drop_duplicates()
print(f"\n删除重复行后: {len(df_no_dup)} 行")

# 基于特定列删除重复行
df_no_dup_col = df.drop_duplicates(subset=['姓名'])
print(f"\n基于'姓名'删除重复行后: {len(df_no_dup_col)} 行")

# 处理异常值
print("\n=== 处理异常值 ===")

# 使用Z-score检测异常值
from scipy import stats

z_scores = np.abs(stats.zscore(df['年龄'].dropna()))
print(f"年龄的Z-score: {z_scores}")

# 标记异常值（Z-score > 2）
outliers = df['年龄'].dropna()[z_scores > 2]
print(f"\n检测到的异常值（Z-score > 2）: {outliers.values}")

# 使用IQR方法检测异常值
Q1 = df['年龄'].quantile(0.25)
Q3 = df['年龄'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"\nIQR方法:")
print(f"Q1 = {Q1}, Q3 = {Q3}, IQR = {IQR}")
print(f"正常范围: [{lower_bound}, {upper_bound}]")

# 替换或删除异常值
df_clean = df.copy()
df_clean.loc[(df_clean['年龄'] < 0) | (df_clean['年龄'] > 100), '年龄'] = np.nan
df_clean['年龄'].fillna(df['年龄'].median(), inplace=True)
print("\n异常值替换为中位数后:")
print(df_clean)

# 数据类型转换
print("\n=== 数据类型转换 ===")

# 查看数据类型
print("数据类型:")
print(df.dtypes)

# 转换日期列
df['入职日期'] = pd.to_datetime(df['入职日期'])
print("\n转换后的数据类型:")
print(df.dtypes)

# 从日期中提取信息
df['入职年'] = df['入职日期'].dt.year
df['入职月'] = df['入职日期'].dt.month
df['入职日'] = df['入职日期'].dt.day
df['入职星期'] = df['入职日期'].dt.dayofweek
df['工作日'] = df['入职日期'].dt.day_name()

print("\n提取的日期信息:")
print(df[['姓名', '入职日期', '入职年', '入职月', '工作日']])

# 转换数值列为字符串
df['年龄_str'] = df['年龄'].astype(str)
print("\n年龄列转换为字符串:")
print(df['年龄_str'].dtype)

# 转换字符串为数值
df['工资_num'] = pd.to_numeric(df['工资'], errors='coerce')
print("\n工资列转换为数值:")
print(df['工资_num'].dtype)

# 字符串数据清洗
print("\n=== 字符串数据清洗 ===")

# 创建包含需要清洗的字符串数据
df_str = pd.DataFrame({
    '城市': ['  北京  ', '上海', 'shanghai', 'BEIJING ', 'Guangzhou  ']
})

print("原始字符串数据:")
print(df_str)

# 去除首尾空白
df_str['城市_clean'] = df_str['城市'].str.strip()
print("\n去除空白后:")
print(df_str['城市_clean'])

# 转换为小写/大写
df_str['城市_lower'] = df_str['城市_clean'].str.lower()
df_str['城市_upper'] = df_str['城市_clean'].str.upper()
print("\n大小写转换:")
print(df_str[['城市_clean', '城市_lower', '城市_upper']])

# 替换字符串
df_str['城市_replace'] = df_str['城市_lower'].str.replace(' ', '')
print("\n替换空格后:")
print(df_str['城市_replace'])

# 规范化数据
print("\n=== 数据规范化 ===")

# 创建示例数据
df_norm = pd.DataFrame({
    '姓名': ['张三', '李四', '王五'],
    '工资': [8000, 12000, 10000],
    '绩效': [0.8, 0.9, 0.7]
})

# 最小-最大规范化
df_norm['工资_norm'] = (df_norm['工资'] - df_norm['工资'].min()) / \
                       (df_norm['工资'].max() - df_norm['工资'].min())

# Z-score标准化
df_norm['工资_zscore'] = (df_norm['工资'] - df_norm['工资'].mean()) / df_norm['工资'].std()

print("规范化后的数据:")
print(df_norm)

# 综合数据清洗流程
print("\n=== 综合数据清洗流程 ===")

def clean_dataframe(df):
    """综合数据清洗函数"""
    df_clean = df.copy()
    
    # 1. 删除完全重复的行
    df_clean = df_clean.drop_duplicates()
    
    # 2. 填充缺失值
    # 数值列用均值填充
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
    
    # 分类列用众数填充
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else 'Unknown', inplace=True)
    
    # 3. 处理异常值
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df_clean.loc[(df_clean[col] < lower) | (df_clean[col] > upper), col] = df_clean[col].median()
    
    return df_clean

# 应用清洗函数
df_final = clean_dataframe(df)
print("清洗后的数据:")
print(df_final)