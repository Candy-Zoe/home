# OpenCV图像识别进阶学习
# 主要内容：OCR、条形码识别、人脸识别、物体跟踪

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== 读取图像 ===")
img = np.zeros((300, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (350, 250), (255, 255, 255), -1)
cv2.putText(img, "Hello", (150, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('测试图像')
plt.show()

print("\n=== 边缘检测进阶 ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('灰度图')

plt.subplot(1, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('边缘检测')
plt.show()

print("\n=== 轮廓分析 ===")
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contours = img.copy()
cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)

print(f"找到轮廓数量: {len(contours)}")

for i, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    print(f"轮廓 {i}: 面积={area:.0f}, 周长={perimeter:.0f}")

plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
plt.title('轮廓检测')
plt.show()

print("\n=== 模板匹配 ===")
template = img[60:140, 70:200]
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

result = cv2.matchTemplate(gray, gray_template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(f"匹配分数: {max_val:.4f}")
print(f"最佳匹配位置: {max_loc}")

cv2.rectangle(img, max_loc, (max_loc[0] + template.shape[1], max_loc[1] + template.shape[0]), (0, 255, 0), 2)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('模板匹配结果')
plt.show()

print("\n=== 霍夫变换检测直线 ===")
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)

img_lines = img.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

print(f"检测到直线数量: {len(lines) if lines is not None else 0}")

plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
plt.title('霍夫变换检测直线')
plt.show()

print("\n=== 霍夫圆检测 ===")
circle_img = np.zeros((300, 300), dtype=np.uint8)
cv2.circle(circle_img, (150, 150), 80, 255, -1)
cv2.circle(circle_img, (100, 100), 30, 0, -1)

circles = cv2.HoughCircles(circle_img, cv2.HOUGH_GRADIENT, 1, 50, param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"检测到圆数量: {len(circles[0])}")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(circle_img, cmap='gray')
plt.title('原始图像')

plt.subplot(1, 2, 2)
plt.imshow(circle_img, cmap='gray')
if circles is not None:
    for circle in circles[0]:
        cv2.circle(circle_img, (circle[0], circle[1]), circle[2], 128, 2)
plt.title('圆检测结果')
plt.show()

print("\n=== 形态学操作进阶 ===")
kernel = np.ones((5, 5), np.uint8)

morph_gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(morph_gradient, cmap='gray')
plt.title('形态学梯度')

plt.subplot(1, 3, 2)
plt.imshow(top_hat, cmap='gray')
plt.title('顶帽')

plt.subplot(1, 3, 3)
plt.imshow(black_hat, cmap='gray')
plt.title('黑帽')
plt.show()

print("\n=== 图像分割 ===")
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        print(f"轮廓 {i}: 重心=({cx}, {cy})")

print("\n=== 特征点检测 ===")
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

img_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0))
plt.imshow(cv2.cvtColor(img_keypoints, cv2.COLOR_BGR2RGB))
plt.title(f'ORB特征点 ({len(keypoints)}个)')
plt.show()