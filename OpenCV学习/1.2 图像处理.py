# OpenCV图像处理学习
# 主要内容：边缘检测、滤波、形态学操作

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), 255, -1)
cv2.circle(img, (100, 100), 30, 0, -1)

plt.imshow(img, cmap='gray')
plt.title('测试图像')
plt.show()

print("\n=== 边缘检测 ===")
edges = cv2.Canny(img, 50, 150)
plt.imshow(edges, cmap='gray')
plt.title('边缘检测')
plt.show()

print("\n=== 滤波操作 ===")
blur = cv2.GaussianBlur(img, (5, 5), 0)
plt.imshow(blur, cmap='gray')
plt.title('高斯模糊')
plt.show()

print("\n=== 形态学操作 ===")
kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)
dilation = cv2.dilate(img, kernel, iterations=1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(erosion, cmap='gray')
plt.title('腐蚀')

plt.subplot(1, 2, 2)
plt.imshow(dilation, cmap='gray')
plt.title('膨胀')
plt.show()

print("\n=== 图像阈值 ===")
_, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap='gray')
plt.title('二值化')
plt.show()