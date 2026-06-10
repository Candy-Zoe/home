# OpenCV图像基础操作学习
# 主要内容：图像读取、显示、保存、颜色空间转换

import cv2
import matplotlib.pyplot as plt

print("=== 读取图像 ===")
img = cv2.imread('example.jpg')
print(f"图像形状: {img.shape}")
print(f"图像类型: {img.dtype}")

print("\n=== 显示图像 ===")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('原始图像')
plt.show()

print("\n=== 颜色空间转换 ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('灰度图像')

plt.subplot(1, 2, 2)
plt.imshow(hsv)
plt.title('HSV图像')
plt.show()

print("\n=== 保存图像 ===")
cv2.imwrite('gray.jpg', gray)
print("灰度图像已保存")

print("\n=== 图像缩放 ===")
resized = cv2.resize(img, (300, 200))
print(f"原始大小: {img.shape}")
print(f"缩放后大小: {resized.shape}")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('gray.jpg'):
    os.remove('gray.jpg')
    print("已删除测试文件")