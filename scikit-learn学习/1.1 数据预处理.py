# scikit-learn数据预处理学习
# 主要内容：数据加载、缺失值处理、特征缩放、类别编码

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

print("=== 创建示例数据集 ===")
data = {
    '年龄': [25, 30, np.nan, 35, 40],
    '收入': [5000, 6000, 7000, np.nan, 9000],
    '城市': ['北京', '上海', '北京', '广州', '上海'],
    '性别': ['男', '女', '男', np.nan, '女']
}
df = pd.DataFrame(data)
print(f"原始数据:\n{df}\n")

print("=== 处理缺失值 ===")
imputer_num = SimpleImputer(strategy='mean')
df[['年龄', '收入']] = imputer_num.fit_transform(df[['年龄', '收入']])

imputer_cat = SimpleImputer(strategy='most_frequent')
df['性别'] = imputer_cat.fit_transform(df[['性别']]).flatten()

print(f"处理后数据:\n{df}\n")

print("=== 特征缩放 - StandardScaler ===")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['年龄', '收入']])
print(f"标准化后:\n{scaled_data}\n")

print("=== 特征缩放 - MinMaxScaler ===")
minmax = MinMaxScaler()
normalized_data = minmax.fit_transform(df[['年龄', '收入']])
print(f"归一化后:\n{normalized_data}\n")

print("=== 类别编码 - OneHotEncoder ===")
encoder = OneHotEncoder(sparse_output=False)
encoded_city = encoder.fit_transform(df[['城市']])
encoded_df = pd.DataFrame(encoded_city, columns=encoder.get_feature_names_out(['城市']))
print(f"独热编码结果:\n{encoded_df}\n")

print("=== 合并处理后的数据 ===")
final_df = pd.concat([df[['年龄', '收入', '性别']], encoded_df], axis=1)
print(f"最终数据:\n{final_df}")