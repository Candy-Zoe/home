# Matplotlib高级图表学习
# 主要内容：多子图、自定义样式、3D图表、动画

# 导入Matplotlib库
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建测试数据
print("=== 创建测试数据 ===")

x = np.linspace(0, 2 * np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.tan(x)

print("测试数据已创建")

# 多子图布局
print("\n=== 多子图布局 ===")

# 方法1：使用subplots创建规则网格
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(x, y1, 'b-')
axes[0, 0].set_title('正弦函数')
axes[0, 0].set_xlabel('X轴')
axes[0, 0].set_ylabel('Y轴')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(x, y2, 'r-')
axes[0, 1].set_title('余弦函数')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].plot(x, y3, 'g-')
axes[1, 0].set_title('正切函数')
axes[1, 0].set_ylim(-5, 5)  # 限制y轴范围
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(x, y1, 'b-', label='sin')
axes[1, 1].plot(x, y2, 'r--', label='cos')
axes[1, 1].set_title('三角函数对比')
axes[1, 1].legend()  # 添加图例
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 方法2：不规则网格布局
print("\n=== 不规则网格布局 ===")

fig = plt.figure(figsize=(14, 6))

# 创建网格规范
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# 大图占两列
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(x, y1, 'b-')
ax1.set_title('大图：正弦函数')

# 小图1
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(x, y2, 'r-')
ax2.set_title('余弦')

# 下方两图
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(x, y1 * y2, 'g-')
ax3.set_title('sin × cos')

ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(x, y1 + y2, 'm-')
ax4.set_title('sin + cos')

ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(x, y1 ** 2 + y2 ** 2, 'k-')
ax5.set_title('sin² + cos² = 1')

plt.suptitle('不规则网格布局示例', fontsize=16)
plt.show()

# 3D图表
print("\n=== 3D图表 ===")

from mpl_toolkits.mplot3d import Axes3D

# 创建3D图形
fig = plt.figure(figsize=(14, 5))

# 3D线图
ax1 = fig.add_subplot(131, projection='3d')
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)
ax1.plot(x, y, z, 'b-', linewidth=2)
ax1.set_title('3D螺旋线')

# 3D散点图
ax2 = fig.add_subplot(132, projection='3d')
n = 100
x = np.random.randn(n)
y = np.random.randn(n)
z = np.random.randn(n)
colors = np.random.rand(n)
ax2.scatter(x, y, z, c=colors, cmap='viridis', alpha=0.6)
ax2.set_title('3D散点图')

# 3D曲面图
ax3 = fig.add_subplot(133, projection='3d')
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))
ax3.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.8)
ax3.set_title('3D曲面图')

plt.tight_layout()
plt.show()

# 极坐标图
print("\n=== 极坐标图 ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw={'projection': 'polar'})

# 极坐标线图
theta = np.linspace(0, 2 * np.pi, 100)
r = 1 + np.sin(theta)
axes[0].plot(theta, r, 'b-', linewidth=2)
axes[0].set_title('极坐标线图 (心形线)')

# 极坐标柱状图
categories = ['A', 'B', 'C', 'D', 'E']
values = [3, 5, 2, 7, 4]
theta = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
theta = np.append(theta, theta[0])
values = np.append(values, values[0])
axes[1].fill(theta, values, alpha=0.5, color='green')
axes[1].plot(theta, values, 'ko-', markersize=8)
axes[1].set_title('极坐标柱状图')

plt.show()

# 自定义样式和颜色
print("\n=== 自定义样式和颜色 ===")

# 使用内置样式
print("可用样式:", plt.style.available)

# 使用seaborn样式
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, y1, color='#FF5733', linewidth=2, linestyle='--', marker='o', markersize=4)
ax.set_title('自定义颜色和线型', fontsize=14)
ax.set_xlabel('X轴（自定义）')
ax.set_ylabel('Y轴（自定义）')
plt.show()

# 恢复默认样式
plt.style.use('default')

# 自定义颜色映射
print("\n=== 自定义颜色映射 ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 颜色条形图
categories = ['类别A', '类别B', '类别C', '类别D', '类别E']
values = [23, 45, 56, 78, 65]
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))

axes[0].bar(categories, values, color=colors)
axes[0].set_title('Viridis颜色映射')
axes[0].set_ylabel('数值')

# 渐变颜色
n = 50
x = np.arange(n)
y = np.random.rand(n)
colors = plt.cm.RdYlGn(y / y.max())

axes[1].scatter(x, y, c=y, cmap='RdYlGn', s=100, edgecolors='black', linewidth=0.5)
axes[1].set_title('RdYlGn颜色映射')
axes[1].set_ylabel('Y值')

plt.colorbar(axes[1].collections[0], ax=axes[1], label='颜色值')
plt.tight_layout()
plt.show()

# 填充区域
print("\n=== 填充区域 ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 基本填充
axes[0, 0].fill(x, y1, alpha=0.3, color='blue')
axes[0, 0].plot(x, y1, 'b-', linewidth=2)
axes[0, 0].set_title('基本填充')

# 多区域填充
axes[0, 1].fill_between(x, y1, y2, alpha=0.5, color='green')
axes[0, 1].plot(x, y1, 'b-', x, y2, 'r--', linewidth=2)
axes[0, 1].set_title('区间填充')

# 带条件的填充
condition = y1 > y2
axes[1, 0].fill_between(x, y1, y2, where=condition, alpha=0.5, color='red')
axes[1, 0].fill_between(x, y1, y2, where=~condition, alpha=0.5, color='blue')
axes[1, 0].set_title('条件填充')

# 堆叠面积图
data = np.random.rand(5, 10)
labels = ['系列A', '系列B', '系列C', '系列D', '系列E']
axes[1, 1].stackplot(x[:10], data, labels=labels, alpha=0.8)
axes[1, 1].set_title('堆叠面积图')
axes[1, 1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# 误差棒图
print("\n=== 误差棒图 ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 简单误差棒
x = np.arange(5)
y = np.random.rand(5) * 10
yerr = np.random.rand(5) * 2

axes[0].errorbar(x, y, yerr=yerr, fmt='o', capsize=5, capthick=2, color='blue')
axes[0].set_title('误差棒图')
axes[0].set_xlabel('X值')
axes[0].set_ylabel('Y值')

# 带不对成误差的误差棒
xerr_lower = np.random.rand(5) * 0.5
xerr_upper = np.random.rand(5) * 1.5
yerr_lower = np.random.rand(5) * 0.5
yerr_upper = np.random.rand(5) * 2

axes[1].errorbar(x, y, xerr=[xerr_lower, xerr_upper], 
                  yerr=[yerr_lower, yerr_upper], fmt='s', capsize=5)
axes[1].set_title('不对称误差棒')

plt.tight_layout()
plt.show()

# 文字和注释
print("\n=== 文字和注释 ===")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(x, y1, 'b-', linewidth=2)

# 添加文字
ax.text(0.5, 0.8, '这是注释文本', transform=ax.transAxes, fontsize=14,
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 添加箭头注释
ax.annotate('峰值点', xy=(np.pi/2, 1), xytext=(np.pi/2+0.5, 0.7),
            fontsize=12, arrowprops=dict(arrowstyle='->', color='red'),
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

ax.set_title('文字和注释示例')
ax.grid(True, alpha=0.3)
plt.show()

# 箱线图
print("\n=== 箱线图 ===")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 简单箱线图
data = [np.random.randn(100) for _ in range(4)]
bp = axes[0].boxplot(data, labels=['A', 'B', 'C', 'D'], patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_title('简单箱线图')

# 带抖动的散点
positions = [1, 2, 3, 4]
for i, pos in enumerate(positions):
    y = data[i]
    x = np.random.normal(pos, 0.05, size=len(y))
    axes[1].scatter(x, y, alpha=0.5, s=20)

axes[1].boxplot(data, positions=positions)
axes[1].set_title('箱线图+散点')

plt.tight_layout()
plt.show()

print("\n高级图表学习完成！")