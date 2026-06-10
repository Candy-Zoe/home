# OpenCV视频处理学习
# 主要内容：视频读取、帧处理、视频保存、摄像头捕获

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试视频 ===")
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test_video.avi', fourcc, 20.0, (640, 480))

for i in range(100):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, f'Frame {i}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    out.write(frame)

out.release()
print("测试视频已创建")

print("\n=== 读取视频 ===")
cap = cv2.VideoCapture('test_video.avi')
print(f"视频帧率: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"视频宽度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
print(f"视频高度: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
print(f"视频帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

print("\n=== 读取帧 ===")
ret, frame = cap.read()
if ret:
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('第一帧')
    plt.show()

print("\n=== 处理视频帧 ===")
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_count += 1

print(f"处理帧数: {frame_count}")
cap.release()

print("\n=== 视频保存 ===")
cap = cv2.VideoCapture('test_video.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    processed = cv2.flip(frame, 1)
    output.write(processed)

cap.release()
output.release()
print("处理后的视频已保存")

print("\n=== 摄像头捕获 ===")
try:
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("摄像头已打开")
        ret, frame = cap.read()
        if ret:
            plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            plt.title('摄像头捕获')
            plt.show()
        cap.release()
    else:
        print("无法打开摄像头")
except Exception as e:
    print(f"摄像头捕获失败: {e}")

print("\n=== 清理测试文件 ===")
import os
for f in ['test_video.avi', 'output_video.avi']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除 {f}")