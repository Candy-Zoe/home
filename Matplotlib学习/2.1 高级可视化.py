# Matplotlib高级可视化学习
# 主要内容：3D可视化、自定义样式、动画、子图布局、高级图表

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

print("=== 3D可视化 ===")
fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(121, projection='3d')
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

ax1.plot_surface(X, Y, Z, cmap='viridis')
ax1.set_title('3D曲面图')

ax2 = fig.add_subplot(122, projection='3d')
X, Y, Z = np.random.randn(3, 100)
ax2.scatter(X, Y, Z, c=Z, cmap='plasma')
ax2.set_title('3D散点图')

plt.tight_layout()
plt.show()

print("\n=== 自定义样式 ===")
plt.style.use('seaborn-v0_8-darkgrid')
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, ax = plt.subplots()
ax.plot(x, y1, label='sin(x)', color='#1f77b4', linewidth=2)
ax.plot(x, y2, label='cos(x)', color='#ff7f0e', linewidth=2)
ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
ax.set_title('自定义样式图表', fontsize=16)
ax.set_xlabel('X轴', fontsize=12)
ax.set_ylabel('Y轴', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n=== 动画 ===")
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
x = np.arange(0, 2*np.pi, 0.01)
line, = ax.plot(x, np.sin(x))

def animate(frame):
    line.set_ydata(np.sin(x + frame/10))
    return line,

ani = FuncAnimation(fig, animate, frames=100, interval=50, blit=True)
print("动画对象已创建")

print("\n=== 复杂子图布局 ===")
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1:])
ax4 = fig.add_subplot(gs[2, :2])
ax5 = fig.add_subplot(gs[2, 2])

ax1.plot(x, np.sin(x))
ax1.set_title('主图')

ax2.bar(['A', 'B', 'C'], [3, 7, 5])
ax2.set_title('柱状图')

ax3.scatter(np.random.randn(50), np.random.randn(50))
ax3.set_title('散点图')

ax4.hist(np.random.randn(100), bins=20)
ax4.set_title('直方图')

ax5.pie([25, 35, 40], labels=['X', 'Y', 'Z'], autopct='%1.1f%%')
ax5.set_title('饼图')

plt.show()

print("\n=== 热力图 ===")
data = np.random.rand(10, 10)
im = plt.imshow(data, cmap='YlOrRd', interpolation='bilinear')

cbar = plt.colorbar(im)
cbar.set_label('数值', rotation=0, labelpad=15)

plt.title('热力图')
plt.show()

print("\n=== 误差棒图 ===")
x = np.arange(0, 10, 1)
y = np.random.randn(10)
yerr = np.abs(np.random.randn(10) * 0.5)

plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, capthick=2, color='blue', ecolor='gray')
plt.title('误差棒图')
plt.show()

print("\n=== 填充区域图 ===")
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.fill_between(x, y1, alpha=0.3, color='blue', label='sin')
plt.fill_between(x, y2, alpha=0.3, color='red', label='cos')
plt.plot(x, y1, x, y2)
plt.legend()
plt.title('填充区域图')
plt.show()

print("\n=== 箭头图 ===")
x = np.arange(0, 3, 0.1)
y = np.sin(x)

plt.quiver(x[:-1], y[:-1], np.diff(x), np.diff(y), scale=1, width=0.005)
plt.title('箭头图')
plt.show()

print("\n=== 极坐标图 ===")
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(121, projection='polar')
theta = np.linspace(0, 2*np.pi, 100)
r = 1 + np.sin(theta)
ax1.plot(theta, r, color='blue')
ax1.fill(theta, r, alpha=0.3, color='blue')
ax1.set_title('极坐标图')

ax2 = fig.add_subplot(122, projection='polar')
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
r = np.random.rand(8)
ax2.bar(theta, r, width=0.5, bottom=0.0)
ax2.set_title('极坐标柱状图')

plt.tight_layout()
plt.show()