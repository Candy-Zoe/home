# Pandas数据聚合与分组学习
# 主要内容：groupby、聚合函数、透视表

import pandas as pd

print("=== 创建示例数据 ===")
data = {
    '部门': ['研发', '研发', '产品', '产品', '销售', '销售', '运营'],
    '姓名': ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九'],
    '年龄': [25, 30, 28, 35, 22, 27, 32],
    '薪资': [8000, 12000, 10000, 15000, 6000, 8000, 9000]
}
df = pd.DataFrame(data)
print(f"原始数据:\n{df}\n")

print("=== 按部门分组 ===")
grouped = df.groupby('部门')
print(f"分组对象: {grouped}")
print(f"\n各组:\n{grouped.groups}")

print("\n=== 分组聚合 ===")
print(f"各部门平均薪资:\n{grouped['薪资'].mean()}")
print(f"\n各部门薪资总和:\n{grouped['薪资'].sum()}")
print(f"\n各部门人数:\n{grouped.size()}")

print("\n=== 多列聚合 ===")
agg_result = grouped.agg({
    '薪资': ['mean', 'sum', 'max'],
    '年龄': ['mean', 'min']
})
print(f"多列聚合结果:\n{agg_result}")

print("\n=== apply方法 ===")
def top_salary(group):
    return group[group['薪资'] == group['薪资'].max()]

top_employees = grouped.apply(top_salary)
print(f"各部门最高薪资员工:\n{top_employees}")

print("\n=== 透视表 ===")
pivot = df.pivot_table(
    index='部门',
    values='薪资',
    aggfunc=['mean', 'sum'],
    margins=True
)
print(f"薪资透视表:\n{pivot}")

print("\n=== 交叉表 ===")
df['级别'] = ['初级', '高级', '中级', '高级', '初级', '中级', '中级']
cross = pd.crosstab(df['部门'], df['级别'])
print(f"部门-级别交叉表:\n{cross}")