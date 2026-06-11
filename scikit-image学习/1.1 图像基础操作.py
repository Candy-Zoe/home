# scikit-image图像处理基础学习
# 主要内容：图像读取、显示、基本操作

# 导入必要的库
from skimage import io, color, filters, transform, util
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
print("=== 读取图像 ===")

# 创建测试图像
test_image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)

# 创建一个简单的测试图像（渐变）
for i in range(200):
    for j in range(200):
        test_image[i, j] = [i, j, (i+j)//2]

print(f"测试图像形状: {test_image.shape}")
print(f"数据类型: {test_image.dtype}")
print(f"像素值范围: [{test_image.min()}, {test_image.max()}]")

# 显示图像
print("\n=== 显示图像 ===")

plt.figure(figsize=(8, 4))

# 显示原始图像
plt.subplot(1, 2, 1)
plt.imshow(test_image)
plt.title('原始图像')
plt.axis('off')

# 显示灰度图像
gray_image = color.rgb2gray(test_image)
plt.subplot(1, 2, 2)
plt.imshow(gray_image, cmap='gray')
plt.title('灰度图像')
plt.axis('off')

plt.tight_layout()
plt.show()

# 图像基本操作
print("\n=== 图像基本操作 ===")

# 图像裁剪
cropped = test_image[50:150, 50:150]
print(f"裁剪后形状: {cropped.shape}")

# 图像缩放
scaled = transform.resize(test_image, (100, 100))
print(f"缩放后形状: {scaled.shape}")

# 图像旋转
rotated = transform.rotate(test_image, 45)
print(f"旋转后形状: {rotated.shape}")

# 图像翻转
flipped_h = np.fliplr(test_image)
flipped_v = np.flipud(test_image)

# 颜色空间转换
print("\n=== 颜色空间转换 ===")

# RGB转HSV
hsv_image = color.rgb2hsv(test_image)
print(f"HSV图像形状: {hsv_image.shape}")

# RGB转Lab
lab_image = color.rgb2lab(test_image)
print(f"Lab图像形状: {lab_image.shape}")

# 可视化颜色空间
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(test_image)
axes[0, 0].set_title('RGB')
axes[0, 0].axis('off')

axes[0, 1].imshow(hsv_image)
axes[0, 1].set_title('HSV')
axes[0, 1].axis('off')

axes[1, 0].imshow(lab_image[:, :, 0], cmap='gray')
axes[1, 0].set_title('Lab - L通道')
axes[1, 0].axis('off')

axes[1, 1].imshow(gray_image, cmap='gray')
axes[1, 1].set_title('灰度')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 图像过滤
print("\n=== 图像过滤 ===")

# 使用不同的滤波器
gaussian = filters.gaussian(gray_image, sigma=2)
median = filters.median(gray_image)
sobel = filters.sobel(gray_image)

# 可视化过滤结果
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].imshow(gray_image, cmap='gray')
axes[0, 0].set_title('原始灰度')
axes[0, 0].axis('off')

axes[0, 1].imshow(gaussian, cmap='gray')
axes[0, 1].set_title('高斯滤波')
axes[0, 1].axis('off')

axes[1, 0].imshow(median, cmap='gray')
axes[1, 0].set_title('中值滤波')
axes[1, 0].axis('off')

axes[1, 1].imshow(sobel, cmap='gray')
axes[1, 1].set_title('Sobel边缘检测')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()

# 图像阈值处理
print("\n=== 图像阈值处理 ===")

# Otsu阈值
threshold = filters.threshold_otsu(gray_image)
binary = gray_image > threshold

# 自适应阈值
local_threshold = filters.threshold_local(gray_image, block_size=31)
binary_local = gray_image > local_threshold

# 可视化阈值处理结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('原始灰度')
axes[0].axis('off')

axes[1].imshow(binary, cmap='gray')
axes[1].set_title(f'Otsu阈值 ({threshold:.2f})')
axes[1].axis('off')

axes[2].imshow(binary_local, cmap='gray')
axes[2].set_title('自适应阈值')
axes[2].axis('off')

plt.tight_layout()
plt.show()

# 图像变换
print("\n=== 图像变换 ===")

# 仿射变换
transform_matrix = transform.AffineTransform(
    scale=(1.5, 1.5),
    rotation=np.pi/6,
    translation=(20, 30)
)
warped = transform.warp(gray_image, transform_matrix)

# 直方图均衡化
equalized = exposure.equalize_hist(gray_image)

# 可视化变换结果
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(warped, cmap='gray')
axes[0].set_title('仿射变换')
axes[0].axis('off')

axes[1].imshow(equalized, cmap='gray')
axes[1].set_title('直方图均衡化')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 图像保存
print("\n=== 图像保存 ===")

# 保存图像
io.imsave('test_output.png', util.img_as_ubyte(test_image))
print("图像已保存为 test_output.png")

# 读取保存的图像
loaded_image = io.imread('test_output.png')
print(f"加载的图像形状: {loaded_image.shape}")

# 清理测试文件
import os
if os.path.exists('test_output.png'):
    os.remove('test_output.png')
    print("测试文件已删除")

print("\nscikit-image基础学习完成！")