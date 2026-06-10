# Pandas数据选择与过滤学习
# 主要内容：条件筛选、布尔索引、query方法

import pandas as pd

print("=== 创建示例数据 ===")
data = {
    '姓名': ['张三', '李四', '王五', '赵六', '钱七'],
    '年龄': [25, 30, 28, 35, 22],
    '城市': ['北京', '上海', '广州', '深圳', '北京'],
    '薪资': [8000, 12000, 10000, 15000, 6000],
    '部门': ['研发', '产品', '销售', '运营', '研发']
}
df = pd.DataFrame(data)
print(f"原始数据:\n{df}\n")

print("=== 条件筛选 ===")
print(f"年龄大于30的员工:\n{df[df['年龄'] > 30]}")

print(f"\n薪资在10000以上的员工:\n{df[df['薪资'] >= 10000]}")

print("\n=== 多条件筛选 ===")
condition = (df['年龄'] < 30) & (df['城市'] == '北京')
print(f"年龄小于30且在北京的员工:\n{df[condition]}")

print("\n=== 使用query方法 ===")
result = df.query('年龄 < 30 and 城市 == "北京"')
print(f"query结果:\n{result}")

print("\n=== isin方法 ===")
cities = ['北京', '上海']
print(f"在北京或上海的员工:\n{df[df['城市'].isin(cities)]}")

print("\n=== str方法 ===")
print(f"姓名以'张'开头的员工:\n{df[df['姓名'].str.startswith('张')]}")

print("\n=== 筛选后修改 ===")
df.loc[df['部门'] == '研发', '薪资'] += 1000
print(f"研发部加薪后:\n{df}")