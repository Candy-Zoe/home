# OpenCV图像分割学习
# 主要内容：阈值分割、边缘检测、区域生长、分水岭算法

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (100, 100, 100), -1)
cv2.rectangle(img, (200, 100), (280, 180), (200, 200, 200), -1)
cv2.circle(img, (150, 250), 30, (150, 150, 150), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始图像')
plt.show()

print("\n=== 简单阈值分割 ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(thresh1, cmap='gray')
plt.title('THRESH_BINARY')

plt.subplot(1, 3, 2)
plt.imshow(thresh2, cmap='gray')
plt.title('THRESH_BINARY_INV')

plt.subplot(1, 3, 3)
plt.imshow(thresh3, cmap='gray')
plt.title('THRESH_TRUNC')
plt.show()

print("\n=== 自适应阈值分割 ===")
thresh_adapt1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
thresh_adapt2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(thresh_adapt1, cmap='gray')
plt.title('ADAPTIVE_THRESH_MEAN_C')

plt.subplot(1, 2, 2)
plt.imshow(thresh_adapt2, cmap='gray')
plt.title('ADAPTIVE_THRESH_GAUSSIAN_C')
plt.show()

print("\n=== Otsu阈值分割 ===")
ret, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f"Otsu阈值: {ret}")

plt.imshow(thresh_otsu, cmap='gray')
plt.title('Otsu阈值分割')
plt.show()

print("\n=== 边缘检测 ===")
edges = cv2.Canny(gray, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title('Canny边缘检测')
plt.show()

print("\n=== 轮廓检测 ===")
contours, hierarchy = cv2.findContours(thresh_otsu, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title('轮廓检测')
plt.show()

print("\n=== 分水岭算法 ===")
img_water = img.copy()
gray = cv2.cvtColor(img_water, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

sure_bg = cv2.dilate(opening, kernel, iterations=3)

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

ret, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1
markers[unknown == 255] = 0

markers = cv2.watershed(img_water, markers)
img_water[markers == -1] = [255, 0, 0]

plt.imshow(cv2.cvtColor(img_water, cv2.COLOR_BGR2RGB))
plt.title('分水岭算法')
plt.show()