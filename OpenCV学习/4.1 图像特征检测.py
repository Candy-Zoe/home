# OpenCV图像特征检测学习
# 主要内容：Harris角点检测、SIFT、SURF、ORB特征提取与匹配

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((300, 300, 3), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), 2)
cv2.rectangle(img, (200, 100), (280, 180), (255, 255, 255), 2)
cv2.circle(img, (100, 250), 30, (255, 255, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('测试图像')
plt.show()

print("\n=== Harris角点检测 ===")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

img_harris = img.copy()
img_harris[dst > 0.01 * dst.max()] = [0, 0, 255]

plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
plt.title('Harris角点检测')
plt.show()

print("\n=== Shi-Tomasi角点检测 ===")
corners = cv2.goodFeaturesToTrack(gray, 10, 0.01, 10)
corners = np.int0(corners)

img_shi = img.copy()
for corner in corners:
    x, y = corner.ravel()
    cv2.circle(img_shi, (x, y), 5, (0, 255, 0), -1)

plt.imshow(cv2.cvtColor(img_shi, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi角点检测')
plt.show()

print("\n=== SIFT特征检测 ===")
sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
print(f"SIFT关键点数量: {len(keypoints)}")

img_sift = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title('SIFT特征')
plt.show()

print("\n=== ORB特征检测 ===")
orb = cv2.ORB_create()
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)
print(f"ORB关键点数量: {len(keypoints_orb)}")

img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0, 255, 0), flags=0)
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.title('ORB特征')
plt.show()

print("\n=== 特征匹配 ===")
img1 = img.copy()
img2 = cv2.flip(img, 1)

kp1, des1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
plt.imshow(cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB))
plt.title('ORB特征匹配')
plt.show()