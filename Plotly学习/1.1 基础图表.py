# Plotly基础图表学习
# 主要内容：散点图、折线图、柱状图、直方图

import plotly.express as px
import pandas as pd

print("=== 加载数据集 ===")
tips = px.data.tips()
print(f"数据集前5行:\n{tips.head()}")

print("\n=== 散点图 ===")
fig = px.scatter(tips, x='total_bill', y='tip', color='sex', title='消费总额与小费关系')
fig.show()

print("\n=== 折线图 ===")
df = pd.DataFrame({
    '年份': [2018, 2019, 2020, 2021, 2022],
    '销售额': [100, 150, 120, 200, 180]
})
fig = px.line(df, x='年份', y='销售额', title='年度销售额趋势')
fig.show()

print("\n=== 柱状图 ===")
fig = px.bar(tips, x='day', y='total_bill', color='sex', barmode='group', title='各天消费总额')
fig.show()

print("\n=== 直方图 ===")
fig = px.histogram(tips, x='total_bill', title='消费总额分布')
fig.show()

print("\n=== 饼图 ===")
fig = px.pie(tips, values='tip', names='day', title='各天小费占比')
fig.show()