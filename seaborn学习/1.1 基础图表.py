# seaborn基础图表学习
# 主要内容：散点图、折线图、柱状图、直方图

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

print("=== 加载数据集 ===")
tips = sns.load_dataset('tips')
print(f"数据集前5行:\n{tips.head()}")

print("\n=== 散点图 ===")
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='sex')
plt.title('消费总额与小费关系')
plt.show()

print("\n=== 折线图 ===")
flights = sns.load_dataset('flights')
sns.lineplot(data=flights, x='year', y='passengers', hue='month')
plt.title('航班乘客数量趋势')
plt.show()

print("\n=== 柱状图 ===")
sns.barplot(data=tips, x='day', y='total_bill', hue='sex')
plt.title('各天消费总额')
plt.show()

print("\n=== 直方图 ===")
sns.histplot(data=tips, x='total_bill', kde=True)
plt.title('消费总额分布')
plt.show()

print("\n=== 箱线图 ===")
sns.boxplot(data=tips, x='day', y='total_bill')
plt.title('各天消费箱线图')
plt.show()