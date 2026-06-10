# OpenCV直方图与匹配学习
# 主要内容：直方图计算、直方图均衡化、直方图匹配、反向投影

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img, (20, 20), (100, 100), (255, 100, 50), -1)
cv2.rectangle(img, (120, 80), (180, 180), (50, 200, 150), -1)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('测试图像')
plt.show()

print("\n=== 计算直方图 ===")
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    hist = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist, color=col)
    plt.xlim([0, 256])

plt.title('RGB直方图')
plt.show()

print("\n=== 灰度直方图 ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

plt.plot(hist)
plt.title('灰度直方图')
plt.show()

print("\n=== 直方图均衡化 ===")
equ = cv2.equalizeHist(gray)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('原始')

plt.subplot(1, 2, 2)
plt.imshow(equ, cmap='gray')
plt.title('均衡化')
plt.show()

print("\n=== CLAHE自适应直方图均衡化 ===")
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('原始')

plt.subplot(1, 2, 2)
plt.imshow(clahe_img, cmap='gray')
plt.title('CLAHE')
plt.show()

print("\n=== 直方图匹配 ===")
img1 = np.zeros((200, 200), dtype=np.uint8)
img1[:100, :] = 100
img1[100:, :] = 200

img2 = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

hist1 = cv2.calcHist([img1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([img2], [0], None, [256], [0, 256])

cdf1 = hist1.cumsum()
cdf2 = hist2.cumsum()

cdf1_norm = cdf1 / cdf1.max()
cdf2_norm = cdf2 / cdf2.max()

lookup_table = np.zeros(256)
cdf1_i = 0
for i in range(256):
    while cdf1_i < 255 and cdf1_norm[cdf1_i] < cdf2_norm[i]:
        cdf1_i += 1
    lookup_table[i] = cdf1_i

matched_img = cv2.LUT(img2, lookup_table.astype(np.uint8))

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(img1, cmap='gray')
plt.title('目标图像')

plt.subplot(1, 3, 2)
plt.imshow(img2, cmap='gray')
plt.title('源图像')

plt.subplot(1, 3, 3)
plt.imshow(matched_img, cmap='gray')
plt.title('匹配后')
plt.show()

print("\n=== 反向投影 ===")
roi = img[20:100, 20:100]
hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

roi_hist = cv2.calcHist([hsv_roi], [0, 1], None, [180, 256], [0, 180, 0, 256])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

dst = cv2.calcBackProject([hsv_img], [0, 1], roi_hist, [0, 180, 0, 256], 1)

disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dst = cv2.filter2D(dst, -1, disc)

ret, thresh = cv2.threshold(dst, 50, 255, 0)
thresh = cv2.merge((thresh, thresh, thresh))
result = cv2.bitwise_and(img, thresh)

plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('反向投影结果')
plt.show()