# OpenCV图像拼接学习
# 主要内容：全景图像拼接、特征匹配、图像融合

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img1 = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img1, (50, 50), (150, 150), (255, 0, 0), 2)
cv2.putText(img1, 'Image 1', (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

img2 = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.rectangle(img2, (50, 50), (150, 150), (0, 255, 0), 2)
cv2.putText(img2, 'Image 2', (60, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
plt.title('图像1')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
plt.title('图像2')
plt.show()

print("\n=== 简单拼接 ===")
h_concat = cv2.hconcat([img1, img2])
v_concat = cv2.vconcat([img1, img2])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(h_concat, cv2.COLOR_BGR2RGB))
plt.title('水平拼接')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(v_concat, cv2.COLOR_BGR2RGB))
plt.title('垂直拼接')
plt.show()

print("\n=== 特征匹配拼接 ===")
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title('特征匹配')
plt.show()

print("\n=== 全景拼接 ===")
try:
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch([img1, img2])
    
    if status == cv2.Stitcher_OK:
        plt.imshow(cv2.cvtColor(stitched, cv2.COLOR_BGR2RGB))
        plt.title('全景拼接')
        plt.show()
    else:
        print(f"拼接失败，状态码: {status}")
except:
    print("OpenCV版本不支持Stitcher")