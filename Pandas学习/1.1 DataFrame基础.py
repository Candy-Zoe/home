# Pandas DataFrame基础学习
# 主要内容：DataFrame的创建、基本属性、查看数据

import pandas as pd
import numpy as np

print("=== 创建DataFrame ===")
data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 28, 35],
    '城市': ['北京', '上海', '广州', '深圳'],
    '薪资': [8000, 12000, 10000, 15000]
}
df = pd.DataFrame(data)
print(f"DataFrame:\n{df}\n")

print("=== 基本属性 ===")
print(f"形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"索引: {df.index.tolist()}")
print(f"数据类型:\n{df.dtypes}\n")

print("=== 查看数据 ===")
print(f"前2行:\n{df.head(2)}")
print(f"\n后2行:\n{df.tail(2)}")
print(f"\n基本统计:\n{df.describe()}")

print("\n=== 访问列 ===")
print(f"姓名列:\n{df['姓名']}")
print(f"\n姓名和年龄:\n{df[['姓名', '年龄']]}")

print("\n=== 访问行 ===")
print(f"第0行:\n{df.iloc[0]}")
print(f"\n前3行:\n{df.iloc[:3]}")

print("\n=== 添加列 ===")
df['部门'] = ['研发', '产品', '销售', '运营']
print(f"添加部门列后:\n{df}")

print("\n=== 修改值 ===")
df.loc[0, '薪资'] = 8500
print(f"修改后:\n{df}")