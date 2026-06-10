# Pandas文件读写学习
# 主要内容：读取CSV、Excel、JSON文件，保存文件

import pandas as pd

print("=== 创建测试数据 ===")
data = {
    '姓名': ['张三', '李四', '王五', '赵六'],
    '年龄': [25, 30, 28, 35],
    '城市': ['北京', '上海', '广州', '深圳'],
    '薪资': [8000, 12000, 10000, 15000]
}
df = pd.DataFrame(data)
print(f"原始数据:\n{df}\n")

print("=== 保存为CSV ===")
df.to_csv('employees.csv', index=False, encoding='utf-8')
print("已保存为 employees.csv")

print("\n=== 读取CSV ===")
df_csv = pd.read_csv('employees.csv')
print(f"从CSV读取:\n{df_csv}")

print("\n=== 保存为Excel ===")
df.to_excel('employees.xlsx', index=False, sheet_name='员工信息')
print("已保存为 employees.xlsx")

print("\n=== 读取Excel ===")
df_excel = pd.read_excel('employees.xlsx', sheet_name='员工信息')
print(f"从Excel读取:\n{df_excel}")

print("\n=== 保存为JSON ===")
df.to_json('employees.json', orient='records', force_ascii=False)
print("已保存为 employees.json")

print("\n=== 读取JSON ===")
df_json = pd.read_json('employees.json')
print(f"从JSON读取:\n{df_json}")

print("\n=== 读取大型CSV ===")
chunk_iter = pd.read_csv('employees.csv', chunksize=2)
for i, chunk in enumerate(chunk_iter):
    print(f"第{i+1}块:\n{chunk}")

print("\n=== 清理测试文件 ===")
import os
for f in ['employees.csv', 'employees.xlsx', 'employees.json']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")