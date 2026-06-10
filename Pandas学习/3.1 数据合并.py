# Pandas数据合并学习
# 主要内容：merge、concat、join

import pandas as pd

print("=== 创建示例数据 ===")
df1 = pd.DataFrame({
    '员工ID': [1, 2, 3, 4],
    '姓名': ['张三', '李四', '王五', '赵六'],
    '部门': ['研发', '产品', '销售', '运营']
})

df2 = pd.DataFrame({
    '员工ID': [1, 2, 3, 5],
    '薪资': [8000, 12000, 10000, 15000],
    '入职日期': ['2023-01-01', '2022-06-15', '2023-03-20', '2024-01-10']
})

print(f"df1:\n{df1}\n")
print(f"df2:\n{df2}\n")

print("=== merge合并 ===")
merged = pd.merge(df1, df2, on='员工ID')
print(f"内连接:\n{merged}")

outer_merge = pd.merge(df1, df2, on='员工ID', how='outer')
print(f"\n外连接:\n{outer_merge}")

left_merge = pd.merge(df1, df2, on='员工ID', how='left')
print(f"\n左连接:\n{left_merge}")

print("\n=== concat拼接 ===")
df3 = pd.DataFrame({
    '员工ID': [6, 7],
    '姓名': ['钱七', '孙八'],
    '部门': ['研发', '产品']
})

concatenated = pd.concat([df1, df3], ignore_index=True)
print(f"纵向拼接:\n{concatenated}")

df4 = pd.DataFrame({
    '绩效': ['A', 'B', 'A', 'C']
}, index=[0, 1, 2, 3])

concat_horizontal = pd.concat([df1, df4], axis=1)
print(f"\n横向拼接:\n{concat_horizontal}")

print("\n=== join方法 ===")
df1.set_index('员工ID', inplace=True)
df2.set_index('员工ID', inplace=True)
joined = df1.join(df2, how='inner')
print(f"join结果:\n{joined}")