# seaborn高级图表学习
# 主要内容：热力图、配对图、联合分布图、小提琴图

import seaborn as sns
import matplotlib.pyplot as plt

print("=== 热力图 ===")
flights = sns.load_dataset('flights')
flights_pivot = flights.pivot('month', 'year', 'passengers')
sns.heatmap(flights_pivot, annot=True, fmt='d', cmap='YlGnBu')
plt.title('航班乘客热力图')
plt.show()

print("\n=== 配对图 ===")
iris = sns.load_dataset('iris')
sns.pairplot(iris, hue='species')
plt.suptitle('鸢尾花特征配对图')
plt.show()

print("\n=== 联合分布图 ===")
tips = sns.load_dataset('tips')
sns.jointplot(data=tips, x='total_bill', y='tip', kind='reg')
plt.suptitle('消费总额与小费联合分布')
plt.show()

print("\n=== 小提琴图 ===")
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex', split=True)
plt.title('各天消费小提琴图')
plt.show()

print("\n=== 计数图 ===")
sns.countplot(data=tips, x='day', hue='sex')
plt.title('各天就餐人数')
plt.show()