# Pandas数据清洗学习
# 主要内容：缺失值处理、重复值处理、数据类型转换

import pandas as pd
import numpy as np

print("=== 创建含缺失值的数据 ===")
data = {
    '姓名': ['张三', '李四', np.nan, '赵六', '钱七'],
    '年龄': [25, np.nan, 28, 35, 22],
    '城市': ['北京', '上海', '广州', np.nan, '北京'],
    '薪资': [8000, 12000, 10000, 15000, np.nan]
}
df = pd.DataFrame(data)
print(f"原始数据:\n{df}\n")

print("=== 检查缺失值 ===")
print(f"缺失值情况:\n{df.isnull()}")
print(f"\n每列缺失值数量:\n{df.isnull().sum()}")

print("\n=== 删除缺失值 ===")
df_dropped = df.dropna()
print(f"删除含缺失值的行后:\n{df_dropped}")

print("\n=== 填充缺失值 ===")
df_filled = df.copy()
df_filled['年龄'] = df_filled['年龄'].fillna(df_filled['年龄'].mean())
df_filled['薪资'] = df_filled['薪资'].fillna(df_filled['薪资'].median())
df_filled['姓名'] = df_filled['姓名'].fillna('未知')
df_filled['城市'] = df_filled['城市'].fillna(method='ffill')
print(f"填充后:\n{df_filled}")

print("\n=== 处理重复值 ===")
df_dup = pd.DataFrame({
    'A': [1, 2, 2, 3, 3, 3],
    'B': ['x', 'y', 'y', 'z', 'z', 'z']
})
print(f"含重复值的数据:\n{df_dup}")
print(f"\n重复行:\n{df_dup.duplicated()}")
df_unique = df_dup.drop_duplicates()
print(f"\n去重后:\n{df_unique}")

print("\n=== 数据类型转换 ===")
df_types = pd.DataFrame({
    '整数': ['1', '2', '3'],
    '浮点数': ['1.5', '2.5', '3.5'],
    '日期': ['2023-01-01', '2023-01-02', '2023-01-03']
})
print(f"原始数据类型:\n{df_types.dtypes}")
df_types['整数'] = df_types['整数'].astype(int)
df_types['浮点数'] = df_types['浮点数'].astype(float)
df_types['日期'] = pd.to_datetime(df_types['日期'])
print(f"\n转换后数据类型:\n{df_types.dtypes}")