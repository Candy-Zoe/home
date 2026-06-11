# seaborn基础图表学习
# 主要内容：分布图、关系图、分类图的绑制

# 导入seaborn库（使用sns作为别名）
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置seaborn样式和配色
print("=== 设置样式 ===")
sns.set_style("whitegrid")  # 设置背景样式
sns.set_palette("husl")     # 设置配色方案
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False

# 生成示例数据
print("\n=== 生成示例数据 ===")

# 创建包含中文的示例数据
np.random.seed(42)
data = pd.DataFrame({
    '姓名': ['张三', '李四', '王五', '赵六', '陈七', '周八', '吴九', '郑十'],
    '年龄': [25, 30, 35, 28, 42, 33, 27, 38],
    '城市': ['北京', '上海', '北京', '深圳', '上海', '北京', '深圳', '上海'],
    '工资': [8000, 12000, 10000, 15000, 9000, 11000, 9500, 13500],
    '绩效': [0.8, 0.9, 0.7, 0.95, 0.6, 0.85, 0.75, 0.88]
})

print("示例数据:")
print(data)

# 分布图
print("\n=== 分布图 ===")

# 创建包含多个子图的画布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 直方图 - 使用displot（新版seaborn使用histplot）
ax1 = axes[0, 0]
sns.histplot(data=data, x='工资', kde=True, ax=ax1)
ax1.set_title('工资分布直方图')
ax1.set_xlabel('工资')
ax1.set_ylabel('频数')

# 2. 核密度估计图
ax2 = axes[0, 1]
sns.kdeplot(data=data, x='工资', ax=ax2, fill=True)
ax2.set_title('工资核密度估计')
ax2.set_xlabel('工资')

# 3. 多变量分布图
ax3 = axes[1, 0]
sns.histplot(data=data, x='工资', y='年龄', ax=ax3, cbar=True)
ax3.set_title('工资-年龄联合分布')
ax3.set_xlabel('工资')
ax3.set_ylabel('年龄')

# 4. 分类变量的分布
ax4 = axes[1, 1]
sns.histplot(data=data, x='工资', hue='城市', kde=True, ax=ax4)
ax4.set_title('不同城市的工资分布')
ax4.set_xlabel('工资')

plt.tight_layout()
plt.show()

# 关系图
print("\n=== 关系图 ===")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 散点图 - 使用relplot
ax1 = axes[0]
sns.scatterplot(data=data, x='工资', y='年龄', hue='城市', 
                size='绩效', sizes=(50, 200), ax=ax1)
ax1.set_title('工资-年龄关系图')
ax1.set_xlabel('工资')
ax1.set_ylabel('年龄')

# 2. 带回归线的散点图
ax2 = axes[1]
sns.regplot(data=data, x='工资', y='年龄', ax=ax2)
ax2.set_title('带回归线的散点图')
ax2.set_xlabel('工资')
ax2.set_ylabel('年龄')

# 3. 折线图
ax3 = axes[2]
sns.lineplot(data=data, x='姓名', y='工资', marker='o', ax=ax3)
ax3.set_title('员工工资折线图')
ax3.set_xlabel('姓名')
ax3.set_ylabel('工资')
ax3.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 分类图
print("\n=== 分类图 ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. 条形图
ax1 = axes[0, 0]
sns.barplot(data=data, x='城市', y='工资', ax=ax1, errorbar='sd')
ax1.set_title('各城市平均工资')
ax1.set_xlabel('城市')
ax1.set_ylabel('平均工资')

# 2. 箱线图
ax2 = axes[0, 1]
sns.boxplot(data=data, x='城市', y='工资', ax=ax2)
ax2.set_title('各城市工资箱线图')
ax2.set_xlabel('城市')
ax2.set_ylabel('工资')

# 3. 小提琴图
ax3 = axes[1, 0]
sns.violinplot(data=data, x='城市', y='工资', ax=ax3)
ax3.set_title('各城市工资小提琴图')
ax3.set_xlabel('城市')
ax3.set_ylabel('工资')

# 4. 点图
ax4 = axes[1, 1]
sns.pointplot(data=data, x='城市', y='工资', ax=ax4, 
              errorbar='sd', color='blue')
ax4.set_title('各城市工资点图')
ax4.set_xlabel('城市')
ax4.set_ylabel('工资')

plt.tight_layout()
plt.show()

# 热力图
print("\n=== 热力图 ===")

# 创建相关矩阵
corr_matrix = data[['年龄', '工资', '绩效']].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
            center=0, fmt='.2f', ax=ax)
ax.set_title('相关性热力图')
plt.tight_layout()
plt.show()

# 配对图
print("\n=== 配对图 ===")

# 选择部分列创建配对图
pair_data = data[['年龄', '工资', '绩效', '城市']]
g = sns.pairplot(pair_data, hue='城市', diag_kind='kde')
g.fig.suptitle('配对关系图', y=1.02)
plt.show()

# FacetGrid多面板图
print("\n=== FacetGrid多面板图 ===")

g = sns.FacetGrid(data, col='城市', height=4, aspect=1)
g.map(sns.histplot, '工资', kde=True)
g.set_axis_labels('工资', '频数')
g.set_titles('{col_name}')
plt.show()