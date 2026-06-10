# Pandas数据透视与聚合学习
# 主要内容：透视表、交叉表、分组聚合、窗口函数

import pandas as pd
import numpy as np

print("=== 创建示例数据 ===")
data = {
    '日期': pd.date_range('2023-01-01', periods=30, freq='D'),
    '类别': np.random.choice(['A', 'B', 'C'], 30),
    '销售额': np.random.randint(100, 500, 30),
    '利润': np.random.randint(10, 100, 30),
    '地区': np.random.choice(['东部', '西部', '南部'], 30)
}
df = pd.DataFrame(data)
print(f"数据前5行:\n{df.head()}")

print("\n=== 分组聚合 ===")
grouped = df.groupby('类别')
print(f"按类别分组的销售额总和:\n{grouped['销售额'].sum()}")
print(f"\n按类别分组的统计:\n{grouped.agg({'销售额': ['sum', 'mean'], '利润': 'max'})}")

print("\n=== 多列分组 ===")
multi_grouped = df.groupby(['类别', '地区'])
print(f"按类别和地区分组的销售额:\n{multi_grouped['销售额'].sum().unstack()}")

print("\n=== 透视表 ===")
pivot = df.pivot_table(values='销售额', index='类别', columns='地区', aggfunc='sum', fill_value=0)
print(f"透视表:\n{pivot}")

print("\n=== 交叉表 ===")
cross = pd.crosstab(df['类别'], df['地区'], values=df['销售额'], aggfunc='sum')
print(f"交叉表:\n{cross}")

print("\n=== 窗口函数 ===")
df['7日滚动平均'] = df['销售额'].rolling(7).mean()
df['累计销售额'] = df['销售额'].cumsum()
print(f"带窗口函数的数据:\n{df[['日期', '销售额', '7日滚动平均', '累计销售额']].head(10)}")

print("\n=== 排名 ===")
df['销售额排名'] = df['销售额'].rank(ascending=False)
print(f"带排名的数据:\n{df[['类别', '销售额', '销售额排名']].head()}")