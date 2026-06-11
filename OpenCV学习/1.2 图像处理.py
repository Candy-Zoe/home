# OpenCV图像处理学习
# 主要内容：图像滤波、边缘检测、形态学操作、图像增强

# 导入OpenCV库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
print("=== 创建测试图像 ===")

# 创建一个包含简单图形的测试图像
img = np.zeros((300, 300, 3), dtype=np.uint8)

# 绘制矩形和圆形
cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
cv2.circle(img, (225, 100), 50, (200, 200, 200), -1)
cv2.rectangle(img, (150, 150), (280, 250), (100, 100, 100), -1)

# 添加一些噪声
noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
img = cv2.add(img, noise)

print("测试图像已创建")

# 图像平滑与滤波
print("\n=== 图像平滑与滤波 ===")

# 均值滤波
blur = cv2.blur(img, (5, 5))
print("均值滤波完成: (5, 5)卷积核")

# 高斯滤波
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
print("高斯滤波完成: (5, 5)卷积核, σ=0")

# 中值滤波
median = cv2.medianBlur(img, 5)
print("中值滤波完成: 卷积核大小=5")

# 双边滤波（保边滤波）
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
print("双边滤波完成: 空间σ=75, 灰度σ=75")

# 显示滤波结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原始图像
axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 均值滤波
axes[0, 1].imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('均值滤波')
axes[0, 1].axis('off')

# 高斯滤波
axes[0, 2].imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('高斯滤波')
axes[0, 2].axis('off')

# 中值滤波
axes[1, 0].imshow(cv2.cvtColor(median, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title('中值滤波')
axes[1, 0].axis('off')

# 双边滤波
axes[1, 1].imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('双边滤波')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 边缘检测
print("\n=== 边缘检测 ===")

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Sobel边缘检测
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.sqrt(sobelx**2 + sobely**2)
print("Sobel边缘检测完成")

# Laplacian边缘检测
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
print("Laplacian边缘检测完成")

# Canny边缘检测
canny = cv2.Canny(gray, 50, 150)
print("Canny边缘检测完成: 阈值1=50, 阈值2=150")

# 显示边缘检测结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('灰度图')
axes[0, 0].axis('off')

axes[0, 1].imshow(sobel, cmap='gray')
axes[0, 1].set_title('Sobel边缘检测')
axes[0, 1].axis('off')

axes[1, 0].imshow(np.abs(laplacian), cmap='gray')
axes[1, 0].set_title('Laplacian边缘检测')
axes[1, 0].axis('off')

axes[1, 1].imshow(canny, cmap='gray')
axes[1, 1].set_title('Canny边缘检测')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 形态学操作
print("\n=== 形态学操作 ===")

# 创建二值图像用于形态学操作
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
print(f"二值图像阈值: 127")

# 腐蚀操作（消除边缘）
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)
print("腐蚀操作完成")

# 膨胀操作（扩大边缘）
dilated = cv2.dilate(binary, kernel, iterations=1)
print("膨胀操作完成")

# 开运算（先腐蚀后膨胀，去除小噪点）
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
print("开运算完成（先腐蚀后膨胀）")

# 闭运算（先膨胀后腐蚀，填充小孔洞）
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
print("闭运算完成（先膨胀后腐蚀）")

# 梯度运算（膨胀图减去腐蚀图，获取边缘）
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
print("梯度运算完成")

# 显示形态学操作结果
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(binary, cmap='gray')
axes[0, 0].set_title('二值图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(eroded, cmap='gray')
axes[0, 1].set_title('腐蚀')
axes[0, 1].axis('off')

axes[0, 2].imshow(dilated, cmap='gray')
axes[0, 2].set_title('膨胀')
axes[0, 2].axis('off')

axes[1, 0].imshow(opened, cmap='gray')
axes[1, 0].set_title('开运算')
axes[1, 0].axis('off')

axes[1, 1].imshow(closed, cmap='gray')
axes[1, 1].set_title('闭运算')
axes[1, 1].axis('off')

axes[1, 2].imshow(gradient, cmap='gray')
axes[1, 2].set_title('梯度')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# 图像增强
print("\n=== 图像增强 ===")

# 对比度增强
alpha = 1.5  # 对比度控制 (1.0-3.0)
beta = 10    # 亮度控制 (0-100)
contrasted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
print(f"对比度增强: alpha={alpha}, beta={beta}")

# 直方图均衡化（用于灰度图）
equalized = cv2.equalizeHist(gray)
print("直方图均衡化完成")

# 自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)
print("自适应直方图均衡化完成: clipLimit=2.0")

# 显示增强结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(contrasted, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'对比度增强 (α={alpha})')
axes[0, 1].axis('off')

axes[1, 0].imshow(equalized, cmap='gray')
axes[1, 0].set_title('直方图均衡化')
axes[1, 0].axis('off')

axes[1, 1].imshow(clahe_img, cmap='gray')
axes[1, 1].set_title('自适应直方图均衡化')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 图像金字塔
print("\n=== 图像金字塔 ===")

# 高斯金字塔
level = 3
img_pyramid = img.copy()
gp = [img_pyramid]

for i in range(level):
    img_pyramid = cv2.pyrDown(img_pyramid)
    gp.append(img_pyramid)
    print(f"高斯金字塔第{i+1}层: {img_pyramid.shape}")

# 拉普拉斯金字塔
lp = [gp[level-1]]

for i in range(level-1, 0, -1):
    size = (gp[i-1].shape[1], gp[i-1].shape[0])
    expanded = cv2.pyrUp(gp[i], dstsize=size)
    laplacian = cv2.subtract(gp[i-1], expanded)
    lp.append(laplacian)
    print(f"拉普拉斯金字塔第{i}层: {laplacian.shape}")

print("图像金字塔构建完成")