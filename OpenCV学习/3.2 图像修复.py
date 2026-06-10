# OpenCV图像修复学习
# 主要内容：图像去噪、图像修复、边缘保持滤波

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始图像')
plt.show()

print("\n=== 添加噪声 ===")
noise = np.random.normal(0, 20, img.shape).astype(np.uint8)
noisy_img = cv2.add(img, noise)

plt.imshow(cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB))
plt.title('带噪声图像')
plt.show()

print("\n=== 高斯滤波去噪 ===")
gaussian_blur = cv2.GaussianBlur(noisy_img, (5, 5), 0)
plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB))
plt.title('高斯滤波')
plt.show()

print("\n=== 中值滤波去噪 ===")
median_blur = cv2.medianBlur(noisy_img, 5)
plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB))
plt.title('中值滤波')
plt.show()

print("\n=== 双边滤波 ===")
bilateral = cv2.bilateralFilter(noisy_img, 9, 75, 75)
plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB))
plt.title('双边滤波')
plt.show()

print("\n=== 创建损坏图像 ===")
damaged_img = img.copy()
cv2.rectangle(damaged_img, (80, 80), (120, 120), (0, 0, 0), -1)

plt.imshow(cv2.cvtColor(damaged_img, cv2.COLOR_BGR2RGB))
plt.title('损坏图像')
plt.show()

print("\n=== 创建修复掩模 ===")
mask = np.zeros(img.shape[:2], dtype=np.uint8)
cv2.rectangle(mask, (80, 80), (120, 120), 255, -1)

plt.imshow(mask, cmap='gray')
plt.title('修复掩模')
plt.show()

print("\n=== 使用INPAINT_NS修复 ===")
inpaint_ns = cv2.inpaint(damaged_img, mask, 3, cv2.INPAINT_NS)
plt.imshow(cv2.cvtColor(inpaint_ns, cv2.COLOR_BGR2RGB))
plt.title('INPAINT_NS修复')
plt.show()

print("\n=== 使用INPAINT_TELEA修复 ===")
inpaint_telea = cv2.inpaint(damaged_img, mask, 3, cv2.INPAINT_TELEA)
plt.imshow(cv2.cvtColor(inpaint_telea, cv2.COLOR_BGR2RGB))
plt.title('INPAINT_TELEA修复')
plt.show()

print("\n=== 边缘检测与增强 ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)

plt.imshow(edges, cmap='gray')
plt.title('边缘检测')
plt.show()

print("\n=== 直方图均衡化 ===")
equalized = cv2.equalizeHist(gray)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('原始')

plt.subplot(1, 2, 2)
plt.imshow(equalized, cmap='gray')
plt.title('均衡化')
plt.show()