# seaborn高级可视化技巧学习
# 主要内容：高级图表、统计图表、自定义样式、多图布局

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

print("=== 加载数据集 ===")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
titanic = sns.load_dataset("titanic")

print("\n=== 联合分布图 ===")
sns.jointplot(data=tips, x="total_bill", y="tip", kind="scatter")
plt.show()

print("\n=== 带回归的联合分布 ===")
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg")
plt.show()

print("\n=== 六边形分箱图 ===")
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex")
plt.show()

print("\n=== 配对图 ===")
sns.pairplot(iris, hue="species")
plt.show()

print("\n=== 热力图 ===")
corr = iris.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('相关性热力图')
plt.show()

print("\n=== 聚类热力图 ===")
sns.clustermap(iris.drop('species', axis=1), cmap='coolwarm', standard_scale=1)
plt.show()

print("\n=== 小提琴图 ===")
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", split=True)
plt.show()

print("\n=== 箱线图 ===")
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker")
plt.show()

print("\n=== 条形图 ===")
sns.barplot(data=titanic, x="sex", y="survived", hue="class")
plt.show()

print("\n=== 计数图 ===")
sns.countplot(data=titanic, x="class", hue="who")
plt.show()

print("\n=== 点图 ===")
sns.pointplot(data=titanic, x="class", y="survived", hue="sex")
plt.show()

print("\n=== 自定义样式 ===")
sns.set_style("whitegrid")
sns.set_context("talk")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
sns.histplot(data=tips, x="total_bill", ax=axes[0])
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time", ax=axes[1])
plt.show()