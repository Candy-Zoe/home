# OpenCV特征检测学习
# 主要内容：关键点检测、特征匹配、人脸识别

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== SIFT特征检测 ===")
img = cv2.imread('example.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
keypoints, descriptors = sift.detectAndCompute(gray, None)
print(f"检测到关键点数量: {len(keypoints)}")

img_sift = cv2.drawKeypoints(img, keypoints, None)
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title('SIFT特征')
plt.show()

print("\n=== SURF特征检测 ===")
try:
    surf = cv2.xfeatures2d.SURF_create()
    keypoints_surf, descriptors_surf = surf.detectAndCompute(gray, None)
    print(f"检测到SURF关键点数量: {len(keypoints_surf)}")
except:
    print("SURF不可用，需要额外安装")

print("\n=== 人脸检测 ===")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

test_img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(test_img, (100, 100), (300, 300), (255, 255, 255), -1)
cv2.circle(test_img, (200, 180), 30, (0, 0, 0), -1)
cv2.circle(test_img, (170, 180), 5, (0, 0, 0), -1)
cv2.circle(test_img, (230, 180), 5, (0, 0, 0), -1)
cv2.ellipse(test_img, (200, 220), (30, 20), 0, 0, 360, (0, 0, 0), 2)

gray_test = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_test, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title('人脸检测')
plt.show()