# OpenCV目标检测学习
# 主要内容：Haar级联检测、人脸检测、眼睛检测

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = np.zeros((400, 400, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
cv2.circle(img, (200, 150), 30, (0, 0, 0), -1)
cv2.circle(img, (165, 150), 5, (0, 0, 0), -1)
cv2.circle(img, (235, 150), 5, (0, 0, 0), -1)
cv2.ellipse(img, (200, 200), (30, 25), 0, 0, 360, (0, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('测试图像')
plt.show()

print("\n=== 人脸检测 ===")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
print(f"检测到人脸数量: {len(faces)}")

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('人脸检测结果')
plt.show()

print("\n=== 眼睛检测 ===")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
eyes = eye_cascade.detectMultiScale(gray)
print(f"检测到眼睛数量: {len(eyes)}")

for (x, y, w, h) in eyes:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('眼睛检测结果')
plt.show()

print("\n=== 微笑检测 ===")
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
smiles = smile_cascade.detectMultiScale(gray, scaleFactor=1.8, minNeighbors=20)
print(f"检测到微笑数量: {len(smiles)}")

for (x, y, w, h) in smiles:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('微笑检测结果')
plt.show()

print("\n=== 身体检测 ===")
body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
bodies = body_cascade.detectMultiScale(gray)
print(f"检测到身体数量: {len(bodies)}")