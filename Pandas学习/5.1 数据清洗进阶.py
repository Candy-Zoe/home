# Pandas数据清洗进阶学习
# 主要内容：缺失值处理策略、数据转换、异常值检测、文本数据处理

import pandas as pd
import numpy as np

print("=== 创建示例数据 ===")
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'age': [25, 32, np.nan, 45, 28, np.nan, 35, 40],
    'income': [50000, 65000, 75000, np.nan, 45000, 80000, np.nan, 90000],
    'gender': ['F', 'M', 'M', 'M', 'F', 'M', 'F', 'M'],
    'city': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris', 'New York', 'London'],
    'score': [85, 90, 78, 92, 88, 75, 95, 80]
}

df = pd.DataFrame(data)
print(f"原始数据:\n{df}")

print("\n=== 缺失值统计 ===")
print(f"缺失值数量:\n{df.isnull().sum()}")
print(f"缺失值比例:\n{df.isnull().mean() * 100}")

print("\n=== 删除缺失值 ===")
df_dropped = df.dropna()
print(f"删除缺失值后:\n{df_dropped}")

print("\n=== 填充缺失值 ===")
df_filled = df.copy()
df_filled['age'] = df_filled['age'].fillna(df_filled['age'].mean())
df_filled['income'] = df_filled['income'].fillna(df_filled['income'].median())
print(f"填充缺失值后:\n{df_filled}")

print("\n=== 插值填充 ===")
df_interpolated = df.copy()
df_interpolated['age'] = df_interpolated['age'].interpolate(method='linear')
print(f"插值填充后:\n{df_interpolated}")

print("\n=== 异常值检测 ===")
z_scores = (df['score'] - df['score'].mean()) / df['score'].std()
outliers = df[np.abs(z_scores) > 2]
print(f"异常值:\n{outliers}")

print("\n=== 数据转换 ===")
df['income_level'] = pd.cut(df['income'], bins=[0, 50000, 70000, 100000], labels=['Low', 'Medium', 'High'])
print(f"添加收入等级后:\n{df}")

print("\n=== 文本数据处理 ===")
df['name_length'] = df['name'].apply(len)
df['name_upper'] = df['name'].str.upper()
print(f"文本处理后:\n{df[['name', 'name_length', 'name_upper']]}")

print("\n=== 数据标准化 ===")
df['score_normalized'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
print(f"标准化分数后:\n{df[['score', 'score_normalized']]}")

print("\n=== 数据分桶 ===")
df['age_group'] = pd.qcut(df['age'], q=3, labels=['Young', 'Middle', 'Old'])
print(f"年龄分组后:\n{df[['age', 'age_group']]}")