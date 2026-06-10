# OpenCV视频处理进阶学习
# 主要内容：视频读写、帧处理、视频特效、实时处理

import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建测试视频 ===")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30
width, height = 640, 480

out = cv2.VideoWriter('test_video.avi', fourcc, fps, (width, height))

for i in range(90):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    x = int(width * i / 90)
    y = int(height / 2 + 100 * np.sin(i * 0.1))
    
    cv2.circle(frame, (x, y), 30, (0, 255, 0), -1)
    cv2.putText(frame, f'Frame {i}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    out.write(frame)

out.release()
print("测试视频已创建 (90帧)")

print("\n=== 读取视频信息 ===")
cap = cv2.VideoCapture('test_video.avi')

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"FPS: {fps}")
print(f"帧数: {frame_count}")
print(f"分辨率: {width}x{height}")

print("\n=== 帧提取 ===")
frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

for i in range(5):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)

cap.release()

plt.figure(figsize=(15, 5))
for i, frame in enumerate(frames):
    plt.subplot(1, 5, i+1)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title(f'帧{i}')
plt.tight_layout()
plt.show()

print("\n=== 视频特效 ===")
cap = cv2.VideoCapture('test_video.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_effect = cv2.VideoWriter('video_effect.avi', fourcc, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    combined = cv2.addWeighted(frame, 0.7, edges_colored, 0.3, 0)
    
    out_effect.write(combined)

cap.release()
out_effect.release()
print("特效视频已创建")

print("\n=== 视频稳定 ===")
cap = cv2.VideoCapture('test_video.avi')
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

transforms = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    dx = np.mean(flow[:, :, 0])
    dy = np.mean(flow[:, :, 1])
    
    transforms.append((dx, dy))
    
    prev_gray = gray

cap.release()
print(f"计算了 {len(transforms)} 帧的运动变换")

print("\n=== 运动检测 ===")
cap = cv2.VideoCapture('test_video.avi')
fgbg = cv2.createBackgroundSubtractorMOG2()

motion_frames = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    fgmask = fgbg.apply(frame)
    
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            motion_detected = True
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    if motion_detected:
        motion_frames.append(frame_idx)
    
    frame_idx += 1

cap.release()
print(f"检测到运动的帧: {motion_frames[:10]}...")

print("\n=== 视频转GIF ===")
import imageio

cap = cv2.VideoCapture('test_video.avi')
frames_gif = []

for _ in range(30):
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_gif.append(frame_rgb)

cap.release()

imageio.mimsave('video.gif', frames_gif, fps=10)
print("GIF已创建")

print("\n=== 清理 ===")
import os
for f in ['test_video.avi', 'video_effect.avi', 'video.gif']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")