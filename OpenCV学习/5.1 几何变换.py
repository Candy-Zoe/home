# OpenCV几何变换学习
# 主要内容：平移、旋转、缩放、仿射变换、透视变换

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
cv2.putText(img, 'Test', (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始图像')
plt.show()

print("\n=== 图像平移 ===")
rows, cols = img.shape[:2]
M = np.float32([[1, 0, 50], [0, 1, 30]])
translated = cv2.warpAffine(img, M, (cols, rows))

plt.imshow(cv2.cvtColor(translated, cv2.COLOR_BGR2RGB))
plt.title('平移')
plt.show()

print("\n=== 图像旋转 ===")
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
rotated = cv2.warpAffine(img, M, (cols, rows))

plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title('旋转45度')
plt.show()

print("\n=== 图像缩放 ===")
scaled_up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
scaled_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(scaled_up, cv2.COLOR_BGR2RGB))
plt.title('放大2倍')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(scaled_down, cv2.COLOR_BGR2RGB))
plt.title('缩小0.5倍')
plt.show()

print("\n=== 仿射变换 ===")
pts1 = np.float32([[50, 50], [150, 50], [50, 150]])
pts2 = np.float32([[50, 70], [130, 40], [70, 150]])

M = cv2.getAffineTransform(pts1, pts2)
affine = cv2.warpAffine(img, M, (cols, rows))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(affine, cv2.COLOR_BGR2RGB))
plt.title('仿射变换')
plt.show()

print("\n=== 透视变换 ===")
pts1 = np.float32([[50, 50], [150, 50], [50, 150], [150, 150]])
pts2 = np.float32([[60, 40], [140, 60], [40, 160], [160, 140]])

M = cv2.getPerspectiveTransform(pts1, pts2)
perspective = cv2.warpPerspective(img, M, (cols, rows))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(perspective, cv2.COLOR_BGR2RGB))
plt.title('透视变换')
plt.show()

print("\n=== 图像翻转 ===")
flip_horizontal = cv2.flip(img, 1)
flip_vertical = cv2.flip(img, 0)
flip_both = cv2.flip(img, -1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(flip_horizontal, cv2.COLOR_BGR2RGB))
plt.title('水平翻转')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(flip_vertical, cv2.COLOR_BGR2RGB))
plt.title('垂直翻转')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(flip_both, cv2.COLOR_BGR2RGB))
plt.title('双向翻转')
plt.show()