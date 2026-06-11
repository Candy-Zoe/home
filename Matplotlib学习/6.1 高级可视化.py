# Matplotlib高级可视化学习
# 主要内容：子图布局、高级图表、颜色映射、自定义样式、3D可视化、动画

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建示例数据
print("=== 创建示例数据 ===")

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.exp(-0.1 * x)
y4 = np.random.randn(100).cumsum()

# 分类数据
categories = ['A', 'B', 'C', 'D', 'E']
values1 = [23, 45, 56, 78, 34]
values2 = [45, 32, 67, 54, 89]

# 散点数据
scatter_x = np.random.randn(200)
scatter_y = np.random.randn(200)
scatter_colors = np.random.rand(200)
scatter_sizes = np.random.rand(200) * 100 + 50

# 2D数据（用于热力图）
X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
Z = np.sin(np.sqrt(X**2 + Y**2))

print("示例数据创建完成")

# 基本绘图回顾
print("\n=== 基本绘图回顾 ===")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
ax.fill_between(x, y1, y2, alpha=0.2, color='green')
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_title('基本绘图示例')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 子图布局 - Subplots
print("\n=== 子图布局 - Subplots ===")

# 2x2网格
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 子图1: 折线图
axes[0, 0].plot(x, y1, 'r-')
axes[0, 0].set_title('sin(x)')
axes[0, 0].grid(True, alpha=0.3)

# 子图2: 散点图
axes[0, 1].scatter(scatter_x, scatter_y, c=scatter_colors, s=scatter_sizes, alpha=0.6, cmap='viridis')
axes[0, 1].set_title('散点图')
axes[0, 1].grid(True, alpha=0.3)

# 子图3: 柱状图
axes[1, 0].bar(categories, values1, color=['red', 'green', 'blue', 'orange', 'purple'])
axes[1, 0].set_title('柱状图')

# 子图4: 直方图
axes[1, 1].hist(y4, bins=20, edgecolor='black', alpha=0.7)
axes[1, 1].set_title('直方图')

plt.suptitle('2x2子图布局', fontsize=14)
plt.tight_layout()
plt.show()

# GridSpec布局
print("\n=== GridSpec高级布局 ===")

fig = plt.figure(figsize=(14, 8))
gs = GridSpec(2, 3, figure=fig)

# 大图 - 跨越2行2列
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax1.plot(x, y1, 'r-', linewidth=2, label='sin(x)')
ax1.plot(x, y2, 'b-', linewidth=2, label='cos(x)')
ax1.set_title('主图 - 跨越2行2列', fontsize=12)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 小图1
ax2 = fig.add_subplot(gs[0, 2])
ax2.scatter(scatter_x, scatter_y, c=scatter_colors, cmap='plasma', alpha=0.6)
ax2.set_title('散点图')

# 小图2
ax3 = fig.add_subplot(gs[1, 2])
ax3.bar(categories, values1, color='steelblue')
ax3.set_title('柱状图')

plt.suptitle('GridSpec布局', fontsize=14)
plt.tight_layout()
plt.show()

# 高级图表类型
print("\n=== 高级图表类型 ===")

# 1. 双Y轴图表
print("1. 双Y轴图表:")
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:blue'
color2 = 'tab:red'

ax1.set_xlabel('X轴')
ax1.set_ylabel('sin(x)', color=color1)
ax1.plot(x, y1, color=color1, linewidth=2)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
ax2.set_ylabel('cos(x) * 2', color=color2)
ax2.plot(x, y2 * 2, color=color2, linewidth=2, linestyle='--')
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('双Y轴图表')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. 面积图
print("\n2. 面积图:")
fig, ax = plt.subplots(figsize=(10, 6))
ax.stackplot(x, y1, y2, y3, labels=['sin(x)', 'cos(x)', 'sin(x)*exp(-0.1x)'],
             colors=['blue', 'green', 'orange'], alpha=0.5)
ax.legend(loc='upper left')
ax.set_title('堆叠面积图')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. 饼图和环形图
print("\n3. 饼图和环形图:")
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 饼图
axes[0].pie(values1, labels=categories, autopct='%1.1f%%', startangle=90,
            colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'])
axes[0].set_title('饼图')

# 环形图
wedges, texts, autotexts = axes[1].pie(values2, labels=categories, autopct='%1.1f%%',
                                        startangle=90, wedgeprops=dict(width=0.3),
                                        colors=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0'])
axes[1].set_title('环形图')

plt.tight_layout()
plt.show()

# 4. 热力图
print("\n4. 热力图:")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 使用imshow
im1 = axes[0].imshow(Z, cmap='hot', extent=[-5, 5, -5, 5], origin='lower')
axes[0].set_title('热力图 - hot colormap')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0])

# 使用pcolormesh
im2 = axes[1].pcolormesh(X, Y, Z, cmap='coolwarm', shading='auto')
axes[1].set_title('热力图 - coolwarm colormap')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# 5. 等高线图
print("\n5. 等高线图:")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 等高线
contour = axes[0].contour(X, Y, Z, cmap='viridis')
axes[0].clabel(contour, inline=True, fontsize=8)
axes[0].set_title('等高线图')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# 填充等高线
contourf = axes[1].contourf(X, Y, Z, cmap='viridis', levels=20)
axes[1].set_title('填充等高线图')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(contourf, ax=axes[1])

plt.tight_layout()
plt.show()

# 6. 小提琴图和箱线图
print("\n6. 小提琴图和箱线图:")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 创建示例数据
data = [np.random.normal(0, std, 100) for std in range(1, 5)]

# 箱线图
axes[0].boxplot(data, showmeans=True, meanline=True)
axes[0].set_title('箱线图')
axes[0].set_xlabel('样本')
axes[0].set_ylabel('值')
axes[0].grid(True, alpha=0.3, axis='y')

# 小提琴图
violin = axes[1].violinplot(data, showmeans=True, showmedians=True)
axes[1].set_title('小提琴图')
axes[1].set_xlabel('样本')
axes[1].set_ylabel('值')
axes[1].grid(True, alpha=0.3, axis='y')

# 设置小提琴颜色
for pc in violin['bodies']:
    pc.set_facecolor('steelblue')
    pc.set_alpha(0.7)

plt.tight_layout()
plt.show()

# 7. 3D可视化
print("\n7. 3D可视化:")

fig = plt.figure(figsize=(18, 5))

# 3D散点图
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(scatter_x, scatter_y, np.random.randn(200), c=scatter_colors, cmap='viridis', alpha=0.6)
ax1.set_title('3D散点图')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# 3D曲面图
ax2 = fig.add_subplot(132, projection='3d')
surf = ax2.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True, alpha=0.8)
ax2.set_title('3D曲面图')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
plt.colorbar(surf, ax=ax2, shrink=0.5)

# 3D线框图
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_wireframe(X, Y, Z, rcount=20, ccount=20, color='steelblue', alpha=0.5)
ax3.set_title('3D线框图')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.show()

# 颜色映射
print("\n=== 颜色映射 (Colormaps) ===")

# 展示不同的colormap
cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'hot', 'cool', 'coolwarm', 'rainbow']
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for ax, cmap_name in zip(axes.flat, cmaps):
    im = ax.imshow(Z, cmap=cmap_name, extent=[-5, 5, -5, 5])
    ax.set_title(cmap_name)
    ax.axis('off')

plt.suptitle('不同颜色映射对比', fontsize=14)
plt.tight_layout()
plt.show()

# 自定义样式和主题
print("\n=== 自定义样式和主题 ===")

# 使用预设样式
styles = ['default', 'seaborn', 'ggplot', 'fivethirtyeight', 'dark_background']

for style in styles:
    plt.style.use(style)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y1, 'r-', linewidth=2)
    ax.plot(x, y2, 'b-', linewidth=2)
    ax.set_title(f'样式: {style}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# 恢复默认样式
plt.style.use('default')

# 自定义颜色和样式
print("\n自定义颜色和样式:")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 不同颜色的线
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']
for i, color in enumerate(colors):
    axes[0, 0].plot(x, np.sin(x + i * 0.5), color=color, linewidth=2, label=f'颜色{i+1}')
axes[0, 0].set_title('自定义颜色')
axes[0, 0].legend(fontsize=8)
axes[0, 0].grid(True, alpha=0.3)

# 不同线型
line_styles = ['-', '--', ':', '-.']
for i, ls in enumerate(line_styles):
    axes[0, 1].plot(x, np.sin(x + i * 0.5), linestyle=ls, linewidth=2, label=ls)
axes[0, 1].set_title('不同线型')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 不同标记
markers = ['o', 's', '^', 'D', 'x']
for i, marker in enumerate(markers):
    axes[1, 0].plot(x[::10], np.sin(x[::10] + i * 0.5), marker=marker, markersize=8, linestyle='', label=marker)
axes[1, 0].set_title('不同标记')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 渐变柱状图
bars = axes[1, 1].bar(categories, values1, color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
axes[1, 1].set_title('渐变柱状图')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 文本和标注
print("\n=== 文本和标注 ===")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y1, 'b-', linewidth=2, label='sin(x)')
ax.plot(x, y3, 'r-', linewidth=2, label='sin(x)*exp(-0.1x)')

# 添加标题和标签
ax.set_title('带有标注的图表', fontsize=14, fontweight='bold')
ax.set_xlabel('X轴', fontsize=12)
ax.set_ylabel('Y轴', fontsize=12)

# 添加文本注释
ax.text(2, 0.5, '这是一段文本', fontsize=10, bbox=dict(facecolor='yellow', alpha=0.5))

# 添加箭头和标注
ax.annotate('峰值', xy=(np.pi/2, 1), xytext=(2, 1.2),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))

ax.annotate('衰减的峰值', xy=(5*np.pi/2, np.sin(5*np.pi/2) * np.exp(-0.1*5*np.pi/2)),
            xytext=(7, 0.5),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5))

# 添加水平线和垂直线
ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax.axvline(x=np.pi, color='green', linestyle=':', alpha=0.5, label='x=π')

# 添加填充区域
ax.fill_between(x, y1, y3, alpha=0.2, color='green')

ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 多面板布局示例
print("\n=== 多面板布局 - 仪表盘示例 ===")

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 主图 - 折线图
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(x, y1, 'r-', linewidth=2, label='sin(x)')
ax_main.plot(x, y2, 'b-', linewidth=2, label='cos(x)')
ax_main.plot(x, y3, 'g-', linewidth=2, label='sin(x)*exp(-0.1x)')
ax_main.set_title('主图 - 多个函数对比', fontsize=12)
ax_main.legend(loc='upper right')
ax_main.grid(True, alpha=0.3)

# 子图1 - 柱状图
ax1 = fig.add_subplot(gs[1, 0])
ax1.bar(categories, values1, color=plt.cm.Set3(np.linspace(0, 1, 5)))
ax1.set_title('柱状图')
ax1.tick_params(axis='x', labelsize=8)

# 子图2 - 饼图
ax2 = fig.add_subplot(gs[1, 1])
ax2.pie(values1, labels=categories, autopct='%1.0f%%', colors=plt.cm.Set2(np.linspace(0, 1, 5)))
ax2.set_title('饼图')

# 子图3 - 散点图
ax3 = fig.add_subplot(gs[1, 2])
ax3.scatter(scatter_x, scatter_y, c=scatter_colors, s=scatter_sizes, alpha=0.6, cmap='viridis')
ax3.set_title('散点图')
ax3.grid(True, alpha=0.3)

# 子图4 - 热力图
ax4 = fig.add_subplot(gs[2, 0:2])
im = ax4.imshow(Z, cmap='coolwarm', extent=[-5, 5, -5, 5], origin='lower')
ax4.set_title('热力图')
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
plt.colorbar(im, ax=ax4)

# 子图5 - 直方图
ax5 = fig.add_subplot(gs[2, 2])
ax5.hist(y4, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
ax5.set_title('直方图')
ax5.grid(True, alpha=0.3, axis='y')

plt.suptitle('仪表盘布局示例', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 动画（简化版）
print("\n=== 动画示例 ===")

# 创建动画数据
x_anim = np.linspace(0, 2*np.pi, 100)

fig, ax = plt.subplots(figsize=(10, 6))
line, = ax.plot([], [], 'r-', linewidth=2)
ax.set_xlim(0, 2*np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_title('简单动画 - sin(x + phase)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.grid(True, alpha=0.3)

def animate(frame):
    """动画函数"""
    phase = frame * 0.1
    y = np.sin(x_anim + phase)
    line.set_data(x_anim, y)
    return line,

# 创建动画（显示前几帧的静态图）
for i in range(5):
    phase = i * 0.5
    y = np.sin(x_anim + phase)
    ax.plot(x_anim, y, alpha=0.3, label=f'phase={phase:.1f}')

ax.legend()
plt.tight_layout()
plt.show()

# 实战案例 - 销售数据分析可视化
print("\n=== 实战案例 - 销售数据分析 ===")

# 创建销售数据
months = ['1月', '2月', '3月', '4月', '5月', '6月', '7月', '8月', '9月', '10月', '11月', '12月']
product_a = np.random.randint(100, 500, 12)
product_b = np.random.randint(80, 400, 12)
product_c = np.random.randint(50, 300, 12)
regions = ['华东', '华北', '华南', '西南']
region_sales = [np.random.randint(500, 1500) for _ in regions]

# 创建多图仪表盘
fig = plt.figure(figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

# 图表1: 月度销售趋势
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(months, product_a, marker='o', linewidth=2, label='产品A')
ax1.plot(months, product_b, marker='s', linewidth=2, label='产品B')
ax1.plot(months, product_c, marker='^', linewidth=2, label='产品C')
ax1.set_title('月度销售趋势', fontsize=12)
ax1.set_xlabel('月份')
ax1.set_ylabel('销量')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# 图表2: 产品销售对比
ax2 = fig.add_subplot(gs[0, 1])
total_sales = [product_a.sum(), product_b.sum(), product_c.sum()]
products = ['产品A', '产品B', '产品C']
bars = ax2.bar(products, total_sales, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax2.set_title('年度产品销售对比', fontsize=12)
ax2.set_ylabel('总销量')
ax2.grid(True, alpha=0.3, axis='y')
# 添加数值标签
for bar, value in zip(bars, total_sales):
    ax2.text(bar.get_x() + bar.get_width()/2, value + 50, f'{value}', ha='center', fontsize=10)

# 图表3: 区域销售分布
ax3 = fig.add_subplot(gs[1, 0])
colors = plt.cm.Set3(np.linspace(0, 1, len(regions)))
wedges, texts, autotexts = ax3.pie(region_sales, labels=regions, autopct='%1.1f%%',
                                     colors=colors, startangle=90)
ax3.set_title('区域销售分布', fontsize=12)

# 图表4: 销售相关性热力图
ax4 = fig.add_subplot(gs[1, 1])
correlation_matrix = np.corrcoef([product_a, product_b, product_c])
im = ax4.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax4.set_xticks(range(3))
ax4.set_yticks(range(3))
ax4.set_xticklabels(['产品A', '产品B', '产品C'])
ax4.set_yticklabels(['产品A', '产品B', '产品C'])
ax4.set_title('产品销售相关性', fontsize=12)
# 添加数值
for i in range(3):
    for j in range(3):
        ax4.text(j, i, f'{correlation_matrix[i, j]:.2f}', ha='center', va='center', color='black')
plt.colorbar(im, ax=ax4)

plt.suptitle('销售数据分析仪表盘', fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# 总结
print("\n=== Matplotlib高级可视化学习总结 ===")
print("1. 子图布局 - subplots, GridSpec")
print("2. 高级图表类型 - 双Y轴、面积图、饼图、环形图")
print("3. 热力图和等高线图 - imshow, pcolormesh, contour")
print("4. 小提琴图和箱线图 - 数据分布可视化")
print("5. 3D可视化 - 散点、曲面、线框图")
print("6. 颜色映射 - colormaps的使用和选择")
print("7. 自定义样式和主题 - style sheets")
print("8. 文本和标注 - 箭头、注释、参考线")
print("9. 仪表盘布局 - 多图组合")
print("10. 动画 - 动态数据可视化")

print("\nMatplotlib高级可视化学习完成！")
