# Matplotlib基础绘图学习
# 主要内容：折线图、散点图、柱状图、直方图、饼图的绘制

# 导入Matplotlib库
import matplotlib.pyplot as plt
import numpy as np

# 准备数据
print("=== 准备数据 ===")

# 生成x轴数据（0到2π，共100个点）
x = np.linspace(0, 2 * np.pi, 100)
# 生成y轴数据（正弦函数）
y_sin = np.sin(x)
# 生成余弦函数数据
y_cos = np.cos(x)

print(f"x数据范围: [{x[0]:.2f}, {x[-1]:.2f}]")
print(f"数据点数: {len(x)}")

# 折线图
print("\n=== 绘制折线图 ===")

plt.figure(figsize=(8, 4))  # 设置画布大小

# 绘制正弦曲线
plt.plot(x, y_sin, label='sin(x)', color='blue', linestyle='-', linewidth=2)
# 绘制余弦曲线
plt.plot(x, y_cos, label='cos(x)', color='red', linestyle='--', linewidth=2)

# 添加标题和标签
plt.title('正弦和余弦函数')
plt.xlabel('x')
plt.ylabel('y')

# 添加图例
plt.legend()

# 添加网格
plt.grid(True, linestyle=':', alpha=0.7)

# 显示图形
plt.show()

# 散点图
print("\n=== 绘制散点图 ===")

# 生成随机数据
np.random.seed(42)
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
sizes = np.random.randint(10, 200, 50)  # 点的大小
colors = np.random.rand(50)  # 点的颜色

plt.figure(figsize=(8, 4))
plt.scatter(x_scatter, y_scatter, s=sizes, c=colors, alpha=0.7, cmap='viridis')
plt.title('随机散点图')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.colorbar(label='颜色值')
plt.show()

# 柱状图
print("\n=== 绘制柱状图 ===")

# 创建数据
categories = ['A', 'B', 'C', 'D', 'E']
values = [30, 45, 25, 50, 35]

plt.figure(figsize=(8, 4))
plt.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
plt.title('类别数据柱状图')
plt.xlabel('类别')
plt.ylabel('数值')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 直方图
print("\n=== 绘制直方图 ===")

# 生成正态分布数据
np.random.seed(42)
data = np.random.randn(1000)

plt.figure(figsize=(8, 4))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.title('正态分布直方图')
plt.xlabel('数值')
plt.ylabel('频数')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# 饼图
print("\n=== 绘制饼图 ===")

# 创建数据
labels = ['苹果', '香蕉', '橙子', '葡萄']
sizes = [35, 25, 20, 20]
explode = (0.1, 0, 0, 0)  # 突出显示第一个部分

plt.figure(figsize=(6, 6))
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('水果销售比例')
plt.axis('equal')  # 保证饼图是圆形
plt.show()

# 子图
print("\n=== 绘制子图 ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 第一个子图：折线图
axes[0, 0].plot(x, y_sin, color='blue')
axes[0, 0].set_title('正弦函数')
axes[0, 0].grid(True)

# 第二个子图：散点图
axes[0, 1].scatter(x_scatter, y_scatter, color='red')
axes[0, 1].set_title('散点图')

# 第三个子图：柱状图
axes[1, 0].bar(categories, values, color='green')
axes[1, 0].set_title('柱状图')

# 第四个子图：直方图
axes[1, 1].hist(data, bins=20, color='orange')
axes[1, 1].set_title('直方图')

plt.tight_layout()
plt.show()