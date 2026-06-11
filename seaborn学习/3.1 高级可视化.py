# seaborn高级可视化学习
# 主要内容：统计绘图、网格图、主题样式、自定义调色板

# 导入必要的库
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 设置样式和主题
print("=== seaborn基础设置 ===")
sns.set_style("whitegrid")  # white, dark, whitegrid, darkgrid, ticks
sns.set_palette("deep")    # deep, muted, pastel, bright, dark, colorblind
plt.rcParams['figure.figsize'] = (10, 6)

# 创建示例数据
print("\n=== 创建示例数据 ===")

# 加载内置数据集
tips = sns.load_dataset("tips")  # 小费数据
iris = sns.load_dataset("iris")  # 鸢尾花数据
flights = sns.load_dataset("flights")  # 航班数据
titanic = sns.load_dataset("titanic")  # 泰坦尼克号数据
diamonds = sns.load_dataset("diamonds")  # 钻石数据

print("数据集信息:")
print(f"  tips: {tips.shape}")
print(f"  iris: {iris.shape}")
print(f"  flights: {flights.shape}")
print(f"  titanic: {titanic.shape}")

# 关系图
print("\n=== 关系图 (relplot) ===")

# 散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 简单散点图
sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[0])
axes[0].set_title('简单散点图')

# 带属性的散点图
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker", 
                style="time", size="size", ax=axes[1])
axes[1].set_title('多属性散点图')

plt.tight_layout()
plt.show()

# 折线图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 简单折线图
sns.lineplot(data=flights, x="year", y="passengers", ax=axes[0])
axes[0].set_title('简单折线图')

# 带置信区间的折线图
flights_mean = flights.groupby('year').mean()
sns.lineplot(data=flights, x="year", y="passengers", 
             hue="month", ax=axes[1])
axes[1].set_title('多系列折线图')

plt.tight_layout()
plt.show()

# 使用relplot创建网格图
print("\n使用relplot创建网格:")
g = sns.relplot(data=tips, x="total_bill", y="tip", 
                col="smoker", row="time", hue="sex", style="day")
g.fig.suptitle('多维度关系图', y=1.02)
plt.tight_layout()
plt.show()

# 分布图
print("\n=== 分布图 (distplot) ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 直方图
sns.histplot(data=tips, x="total_bill", kde=True, ax=axes[0, 0])
axes[0, 0].set_title('直方图 + KDE')

# 按类别分组
sns.histplot(data=tips, x="total_bill", hue="sex", kde=True, ax=axes[0, 1])
axes[0, 1].set_title('分组直方图')

# 核密度估计
sns.kdeplot(data=tips, x="total_bill", ax=axes[1, 0])
axes[1, 0].set_title('核密度估计')

# 2D KDE
sns.kdeplot(data=tips, x="total_bill", y="tip", ax=axes[1, 1])
axes[1, 1].set_title('2D核密度估计')

plt.tight_layout()
plt.show()

# 箱线图和小提琴图
print("\n=== 箱线图和小提琴图 ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 简单箱线图
sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title('简单箱线图')

# 带色调的箱线图
sns.boxplot(data=tips, x="day", y="total_bill", hue="smoker", ax=axes[0, 1])
axes[0, 1].set_title('分组箱线图')

# 小提琴图
sns.violinplot(data=tips, x="day", y="total_bill", ax=axes[1, 0])
axes[1, 0].set_title('小提琴图')

# 带色调的小提琴图
sns.violinplot(data=tips, x="day", y="total_bill", hue="sex", 
               split=True, ax=axes[1, 1])
axes[1, 1].set_title('拆分小提琴图')

plt.tight_layout()
plt.show()

# 条形图和计数图
print("\n=== 条形图和计数图 ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 简单条形图
sns.barplot(data=tips, x="day", y="total_bill", ax=axes[0, 0])
axes[0, 0].set_title('简单条形图')

# 带误差棒的条形图
sns.barplot(data=tips, x="day", y="total_bill", hue="sex", ax=axes[0, 1])
axes[0, 1].set_title('分组条形图')

# 计数图
sns.countplot(data=tips, x="day", ax=axes[1, 0])
axes[1, 0].set_title('计数图')

# 点图
sns.pointplot(data=tips, x="day", y="total_bill", hue="smoker", ax=axes[1, 1])
axes[1, 1].set_title('点图')

plt.tight_layout()
plt.show()

# 热力图
print("\n=== 热力图 ===")

# 创建数据
flights_pivot = flights.pivot_table(index='month', columns='year', values='passengers')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 基础热力图
sns.heatmap(flights_pivot, annot=True, fmt='d', cmap='YlGnBu', ax=axes[0])
axes[0].set_title('基础热力图')

# 带掩码的热力图
mask = flights_pivot < 300
sns.heatmap(flights_pivot, mask=mask, cmap='coolwarm', ax=axes[1])
axes[1].set_title('带掩码热力图')

# 带中心值的热力图
sns.heatmap(flights_pivot, center=flights_pivot.loc['June', 1955], 
            cmap='RdBu_r', ax=axes[2])
axes[2].set_title('带中心值热力图')

plt.tight_layout()
plt.show()

# 相关性热力图
print("\n相关性热力图:")
corr = iris.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('相关性热力图')
plt.tight_layout()
plt.show()

# 分类图
print("\n=== 分类图 (catplot) ===")

# 使用catplot创建网格分类图
g = sns.catplot(data=tips, x="day", y="total_bill", 
                col="smoker", hue="sex", kind="box")
g.fig.suptitle('多维度分类图', y=1.02)
plt.tight_layout()
plt.show()

# 回归图
print("\n=== 回归图 (regplot) ===")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 简单回归图
sns.regplot(data=tips, x="total_bill", y="tip", ax=axes[0, 0])
axes[0, 0].set_title('简单回归图')

# 带置信区间
sns.regplot(data=tips, x="total_bill", y="tip", 
            ci=95, ax=axes[0, 1])
axes[0, 1].set_title('95%置信区间')

# 低ess拟合
sns.regplot(data=tips, x="total_bill", y="tip", lowess=True, ax=axes[1, 0])
axes[1, 0].set_title('LOWESS平滑')

# 按类别分组
sns.regplot(data=tips[tips['smoker']=='Yes'], x="total_bill", y="tip", 
            label='吸烟者', ax=axes[1, 1])
sns.regplot(data=tips[tips['smoker']=='No'], x="total_bill", y="tip", 
            label='非吸烟者', ax=axes[1, 1])
axes[1, 1].legend()
axes[1, 1].set_title('分组回归图')

plt.tight_layout()
plt.show()

# 使用lmplot创建网格回归图
print("\n使用lmplot创建网格:")
g = sns.lmplot(data=tips, x="total_bill", y="tip", col="smoker", row="time")
g.fig.suptitle('多维度回归图', y=1.02)
plt.tight_layout()
plt.show()

# PairGrid和PairPlot
print("\n=== PairGrid和PairPlot ===")

# PairPlot（简化版）
g = sns.pairplot(iris, hue="species", diag_kind="hist", 
                  palette="husl", markers=["o", "s", "D"])
g.fig.suptitle('配对图', y=1.02)
plt.tight_layout()
plt.show()

# PairGrid（自定义）
print("\n自定义PairGrid:")
g = sns.PairGrid(iris, hue="species", diag_sharey=False)
g.map_diag(sns.histplot, kde=True)
g.map_offdiag(sns.scatterplot, alpha=0.6)
g.add_legend()
g.fig.suptitle('自定义配对网格', y=1.02)
plt.tight_layout()
plt.show()

# JointGrid和JointPlot
print("\n=== JointGrid和JointPlot ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# JointPlot
sns.jointplot(data=tips, x="total_bill", y="tip", kind="scatter", ax=axes[0])
axes[0].set_title('散点联合图')

# 带回归的联合图
sns.jointplot(data=tips, x="total_bill", y="tip", kind="reg", ax=axes[1])
axes[1].set_title('回归联合图')

plt.tight_layout()
plt.show()

# 六边形图和KDE联合图
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 六边形图
sns.jointplot(data=tips, x="total_bill", y="tip", kind="hex", ax=axes[0])
axes[0].set_title('六边形联合图')

# KDE联合图
sns.jointplot(data=tips, x="total_bill", y="tip", kind="kde", ax=axes[1])
axes[1].set_title('KDE联合图')

plt.tight_layout()
plt.show()

# FacetGrid
print("\n=== FacetGrid ===")

g = sns.FacetGrid(tips, col="time", row="smoker", height=3, aspect=1.3)
g.map(sns.histplot, "total_bill", kde=True)
g.add_legend()
g.fig.suptitle('FacetGrid', y=1.02)
plt.tight_layout()
plt.show()

# 复杂网格
print("\n复杂网格:")
g = sns.FacetGrid(tips, col="time", row="smoker", height=3, aspect=1.3, 
                   hue="sex", margin_titles=True)
g.map_dataframe(sns.scatterplot, x="total_bill", y="tip", alpha=0.6)
g.map_dataframe(sns.regplot, x="total_bill", y="tip", scatter=False)
g.add_legend()
g.fig.suptitle('复杂FacetGrid', y=1.02)
plt.tight_layout()
plt.show()

# 主题和样式
print("\n=== 主题和样式 ===}")

# 可用的样式
styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, style in enumerate(styles):
    sns.set_style(style)
    sns.boxplot(data=tips, x="day", y="total_bill", ax=axes[idx])
    axes[idx].set_title(f'样式: {style}')

# 隐藏最后一个空白子图
axes[5].axis('off')

# 恢复默认样式
sns.set_theme()

plt.suptitle('seaborn样式示例', fontsize=14)
plt.tight_layout()
plt.show()

# 调色板
print("\n=== 调色板 ===")

# 获取调色板
palettes = ['deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind', 
            'husl', 'Set1', 'Set2', 'Set3', 'Paired', 'tab10']

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, palette in enumerate(palettes):
    sns.set_palette(palette)
    colors = sns.color_palette()
    sns.barplot(x=list(range(10)), y=list(range(10)), 
                palette=palette, ax=axes[idx])
    axes[idx].set_title(f'调色板: {palette}')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

# 恢复默认调色板
sns.set_palette("deep")

plt.suptitle('seaborn调色板', fontsize=14)
plt.tight_layout()
plt.show()

# 自定义调色板
print("\n自定义调色板:")
custom_palette = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
sns.set_palette(custom_palette)

plt.figure(figsize=(10, 4))
sns.barplot(x=list(range(5)), y=list(range(5)), palette=custom_palette)
plt.title('自定义调色板')
plt.tight_layout()
plt.show()

# 圆形调色板
print("\n圆形调色板:")
sns.set_palette("husl", 8)
plt.figure(figsize=(10, 4))
for i, color in enumerate(sns.color_palette("husl", 8)):
    plt.bar(i, 1, color=color)
plt.title('圆形调色板 (husl)')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.show()

# 渐变调色板
print("\n渐变调色板:")
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

gradients = ['Blues', 'Greens', 'Reds', 'Purples']
for idx, gradient in enumerate(gradients):
    sns.barplot(x=list(range(10)), y=list(range(10)),
                palette=gradient, ax=axes[idx])
    axes[idx].set_title(f'渐变: {gradient}')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

plt.suptitle('渐变调色板', fontsize=14)
plt.tight_layout()
plt.show()

# 上下文设置
print("\n=== 上下文设置 ===")

contexts = ['paper', 'notebook', 'talk', 'poster']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, context in enumerate(contexts):
    sns.set_context(context)
    sns.scatterplot(data=tips, x="total_bill", y="tip", ax=axes[idx])
    axes[idx].set_title(f'上下文: {context}')

# 恢复默认上下文
sns.set_context("notebook")

plt.suptitle('seaborn上下文设置', fontsize=14)
plt.tight_layout()
plt.show()

# 复杂可视化示例
print("\n=== 复杂可视化示例 ===")

# 创建综合仪表板
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 主要散点图
ax1 = fig.add_subplot(gs[0, :2])
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", 
                style="smoker", size="size", alpha=0.7, ax=ax1)
ax1.set_title('主要散点图')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# 2. 分布图
ax2 = fig.add_subplot(gs[0, 2])
sns.histplot(data=tips, x="total_bill", kde=True, ax=ax2)
ax2.set_title('账单分布')

# 3. 箱线图
ax3 = fig.add_subplot(gs[1, 0])
sns.boxplot(data=tips, x="day", y="total_bill", ax=ax3)
ax3.set_title('按天账单分布')

# 4. 小提琴图
ax4 = fig.add_subplot(gs[1, 1])
sns.violinplot(data=tips, x="day", y="tip", ax=ax4)
ax4.set_title('按天小费分布')

# 5. 热力图
ax5 = fig.add_subplot(gs[1, 2])
day_smoker = tips.groupby(['day', 'smoker'])['total_bill'].mean().unstack()
sns.heatmap(day_smoker, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax5)
ax5.set_title('按天和吸烟状态的平均账单')

# 6. 计数图
ax6 = fig.add_subplot(gs[2, 0])
sns.countplot(data=tips, x="day", hue="time", ax=ax6)
ax6.set_title('按天和时间的计数')

# 7. 条形图
ax7 = fig.add_subplot(gs[2, 1])
sns.barplot(data=tips, x="sex", y="tip", hue="smoker", ax=ax7)
ax7.set_title('按性别和吸烟状态的小费')

# 8. 点图
ax8 = fig.add_subplot(gs[2, 2])
sns.pointplot(data=tips, x="day", y="tip", hue="sex", ax=ax8)
ax8.set_title('按天和性别的平均小费')

plt.suptitle('seaborn综合仪表板', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 统计注释
print("\n=== 统计注释 ===")

# 添加统计注释
plt.figure(figsize=(10, 6))
ax = sns.boxplot(data=tips, x="day", y="total_bill")

# 计算并显示统计信息
for i, day in enumerate(['Thur', 'Fri', 'Sat', 'Sun']):
    day_data = tips[tips['day'] == day]['total_bill']
    mean = day_data.mean()
    std = day_data.std()
    ax.annotate(f'μ={mean:.1f}\nσ={std:.1f}', 
                xy=(i, day_data.max()), 
                ha='center', va='bottom',
                fontsize=9, color='red')

plt.title('带统计注释的箱线图')
plt.tight_layout()
plt.show()

# 总结
print("\n=== seaborn高级可视化学习总结 ===")
print("1. 关系图 (scatter, line, relplot)")
print("2. 分布图 (histplot, kdeplot)")
print("3. 分类图 (boxplot, violinplot, barplot, countplot)")
print("4. 回归图 (regplot, lmplot)")
print("5. 热力图 (heatmap)")
print("6. 网格图 (PairGrid, PairPlot, FacetGrid, JointGrid)")
print("7. 主题和样式设置")
print("8. 自定义调色板")
print("9. 上下文设置")
print("10. 复杂可视化组合")

print("\nseaborn高级可视化学习完成！")
