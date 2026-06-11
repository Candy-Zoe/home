# OpenCV视频处理与分析学习
# 主要内容：视频读取、视频写入、帧处理、运动检测、视频跟踪

# 导入必要的库
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# 创建视频写入器
print("=== 创建视频写入器 ===")

# 定义视频参数
width = 640
height = 480
fps = 24
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))

print(f"视频写入器创建完成")
print(f"分辨率: {width}x{height}")
print(f"帧率: {fps} FPS")
print(f"编码器: XVID")

# 创建示例视频帧
print("\n=== 创建示例视频帧 ===")

# 创建一些示例帧
for frame_idx in range(100):
    # 创建黑色背景
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 添加动态元素
    # 移动的圆形
    x = int(width/2 + 200 * np.sin(frame_idx * 0.1))
    y = int(height/2 + 150 * np.cos(frame_idx * 0.15))
    radius = int(20 + 10 * np.sin(frame_idx * 0.05))
    cv2.circle(frame, (x, y), radius, (0, 255, 0), -1)
    
    # 添加文字
    cv2.putText(frame, f'Frame: {frame_idx+1}', (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 添加时间信息
    cv2.putText(frame, time.strftime('%H:%M:%S'), (width-150, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 添加进度条
    progress = frame_idx / 100
    cv2.rectangle(frame, (20, height-30), (int(20 + 600*progress), height-10), 
                  (0, 255, 0), -1)
    cv2.putText(frame, f'{int(progress*100)}%', (width-60, height-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # 写入帧
    out.write(frame)
    
    # 显示进度
    if (frame_idx + 1) % 20 == 0:
        print(f"已写入 {frame_idx + 1}/100 帧")

# 释放视频写入器
out.release()
print("视频写入完成")

# 读取视频文件
print("\n=== 读取视频文件 ===")

cap = cv2.VideoCapture('output.avi')

if not cap.isOpened():
    print("无法打开视频文件")
else:
    print("视频文件打开成功")
    print(f"视频宽度: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"视频高度: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"帧率: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"总帧数: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

# 读取并显示视频帧
print("\n读取视频帧...")
frames = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()
print(f"读取到 {len(frames)} 帧")

# 显示第一帧和最后一帧
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
axes[0].set_title('第一帧')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(frames[-1], cv2.COLOR_BGR2RGB))
axes[1].set_title('最后一帧')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 运动检测
print("\n=== 运动检测 ===")

# 使用背景减除方法
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

# 创建结果视频
out_motion = cv2.VideoWriter('motion_output.avi', fourcc, fps, (width, height))

# 处理每一帧
motion_frames = []
for frame in frames:
    # 应用背景减除
    fgmask = fgbg.apply(frame)
    
    # 去除阴影（阴影用灰色表示）
    fgmask[fgmask == 127] = 0
    
    # 形态学操作去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制运动区域
    frame_with_motion = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_motion, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    motion_frames.append(frame_with_motion)
    out_motion.write(frame_with_motion)

out_motion.release()
print("运动检测完成")

# 显示运动检测结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(frames[50], cv2.COLOR_BGR2RGB))
axes[0].set_title('原始帧')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(motion_frames[50], cv2.COLOR_BGR2RGB))
axes[1].set_title('运动检测结果')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 帧差分运动检测
print("\n=== 帧差分运动检测 ===")

# 初始化前一帧
prev_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

frame_diff_frames = []
for i, frame in enumerate(frames[1:], 1):
    # 当前帧转换为灰度
    curr_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 计算帧差分
    frame_diff = cv2.absdiff(prev_frame, curr_frame)
    
    # 二值化
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
    
    # 形态学操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    
    # 寻找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 绘制运动区域
    frame_with_diff = frame.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame_with_diff, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    frame_diff_frames.append(frame_with_diff)
    
    # 更新前一帧
    prev_frame = curr_frame

# 显示帧差分结果
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(frames[50], cv2.COLOR_BGR2RGB))
axes[0].set_title('原始帧')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(frame_diff_frames[49], cv2.COLOR_BGR2RGB))
axes[1].set_title('帧差分结果')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# 对象跟踪
print("\n=== 对象跟踪 ===")

# 选择一个ROI进行跟踪
frame = frames[0]
roi = (280, 190, 80, 80)  # (x, y, w, h)

# 创建跟踪器
tracker = cv2.TrackerCSRT_create()

# 初始化跟踪器
ok = tracker.init(frame, roi)

# 跟踪结果
tracked_frames = []
for frame in frames:
    # 更新跟踪器
    ok, bbox = tracker.update(frame)
    
    if ok:
        # 跟踪成功
        (x, y, w, h) = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, 'Tracking', (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    else:
        # 跟踪失败
        cv2.putText(frame, 'Lost', (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    tracked_frames.append(frame)

# 显示跟踪结果
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(cv2.cvtColor(tracked_frames[0], cv2.COLOR_BGR2RGB))
axes[0].set_title('第1帧')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(tracked_frames[50], cv2.COLOR_BGR2RGB))
axes[1].set_title('第51帧')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(tracked_frames[-1], cv2.COLOR_BGR2RGB))
axes[2].set_title('最后一帧')
axes[2].axis('off')

plt.suptitle('对象跟踪结果')
plt.tight_layout()
plt.show()

# 不同跟踪器对比
print("\n=== 不同跟踪器对比 ===")

tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

# 创建不同跟踪器
def create_tracker(tracker_type):
    if tracker_type == 'BOOSTING':
        return cv2.TrackerBoosting_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    elif tracker_type == 'KCF':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'TLD':
        return cv2.TrackerTLD_create()
    elif tracker_type == 'MEDIANFLOW':
        return cv2.TrackerMedianFlow_create()
    elif tracker_type == 'GOTURN':
        return cv2.TrackerGOTURN_create()
    elif tracker_type == 'MOSSE':
        return cv2.TrackerMOSSE_create()
    elif tracker_type == 'CSRT':
        return cv2.TrackerCSRT_create()
    else:
        return None

# 测试不同跟踪器
print("跟踪器性能对比:")
for tracker_type in tracker_types[:4]:  # 只测试前4个
    tracker = create_tracker(tracker_type)
    if tracker is None:
        continue
    
    tracker.init(frames[0], roi)
    success_count = 0
    
    for frame in frames:
        ok, _ = tracker.update(frame)
        if ok:
            success_count += 1
    
    success_rate = success_count / len(frames) * 100
    print(f"  {tracker_type}: {success_rate:.1f}%")

# 视频帧处理流水线
print("\n=== 视频帧处理流水线 ===")

def process_frame(frame):
    """处理单帧的流水线"""
    # 1. 转换为灰度
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 2. 高斯模糊
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Canny边缘检测
    edges = cv2.Canny(blur, 50, 150)
    
    # 4. 寻找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 5. 绘制轮廓
    result = frame.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    return result

# 处理所有帧
processed_frames = []
for frame in frames:
    processed_frames.append(process_frame(frame))

# 显示处理结果
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(cv2.cvtColor(frames[50], cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始帧')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(cv2.cvtColor(frames[50], cv2.COLOR_BGR2GRAY), cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('灰度图')
axes[0, 1].axis('off')

axes[1, 0].imshow(cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(frames[50], cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150), cmap='gray')
axes[1, 0].set_title('边缘检测')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(processed_frames[50], cv2.COLOR_BGR2RGB))
axes[1, 1].set_title('轮廓提取')
axes[1, 1].axis('off')

plt.suptitle('视频帧处理流水线')
plt.tight_layout()
plt.show()

# 视频质量分析
print("\n=== 视频质量分析 ===")

# 计算帧差异
frame_diffs = []
for i in range(len(frames)-1):
    diff = cv2.absdiff(frames[i], frames[i+1])
    avg_diff = np.mean(diff)
    frame_diffs.append(avg_diff)

# 绘制帧差异曲线
plt.figure(figsize=(10, 5))
plt.plot(frame_diffs)
plt.xlabel('帧索引')
plt.ylabel('平均像素差异')
plt.title('帧间差异变化')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 计算平均亮度
brightness_values = []
for frame in frames:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    brightness_values.append(brightness)

plt.figure(figsize=(10, 5))
plt.plot(brightness_values)
plt.xlabel('帧索引')
plt.ylabel('平均亮度')
plt.title('帧亮度变化')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 清理临时文件
import os
for f in ['output.avi', 'motion_output.avi']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除临时文件: {f}")

print("\nOpenCV视频处理与分析学习完成！")
