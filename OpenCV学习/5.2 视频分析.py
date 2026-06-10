# OpenCV视频分析学习
# 主要内容：光流法、背景减除、运动检测、目标跟踪

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试视频 ===")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('motion_test.avi', fourcc, 10.0, (300, 200))

for i in range(30):
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    x = 50 + i * 5
    cv2.circle(frame, (x, 100), 20, (0, 255, 0), -1)
    out.write(frame)

out.release()
print("测试视频已创建")

print("\n=== 背景减除 ===")
cap = cv2.VideoCapture('motion_test.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()

frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    fgmask = fgbg.apply(frame)
    frames.append(fgmask)

cap.release()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(frames[5], cmap='gray')
plt.title('第5帧')

plt.subplot(1, 3, 2)
plt.imshow(frames[15], cmap='gray')
plt.title('第15帧')

plt.subplot(1, 3, 3)
plt.imshow(frames[25], cmap='gray')
plt.title('第25帧')
plt.show()

print("\n=== 光流法 ===")
cap = cv2.VideoCapture('motion_test.avi')

ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break
    
    next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    prvs = next_frame

cap.release()

plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
plt.title('光流可视化')
plt.show()

print("\n=== Lucas-Kanade光流 ===")
cap = cv2.VideoCapture('motion_test.avi')

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)

color = np.random.randint(0, 255, (100, 3))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    
    img = cv2.add(frame, mask)
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('Lucas-Kanade光流')
plt.show()

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('motion_test.avi'):
    os.remove('motion_test.avi')
    print("已删除测试视频")