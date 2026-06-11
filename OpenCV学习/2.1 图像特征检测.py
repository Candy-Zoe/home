# OpenCV特征检测与匹配学习
# 主要内容：角点检测、边缘检测、特征描述符、特征匹配

# 导入OpenCV库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 创建测试图像
print("=== 创建测试图像 ===")

# 创建一个带有角点的测试图像
img = np.zeros((400, 600, 3), dtype=np.uint8)
img.fill(255)

# 绘制一个矩形
cv2.rectangle(img, (50, 50), (250, 200), (0, 0, 0), 2)

# 绘制一个三角形
pts = np.array([[400, 50], [500, 200], [300, 200]], np.int32)
cv2.polylines(img, [pts], True, (0, 0, 0), 2)

# 绘制圆形
cv2.circle(img, (150, 300), 50, (0, 0, 0), 2)

# 绘制线条
cv2.line(img, (300, 250), (550, 250), (0, 0, 0), 2)

# 添加一些文字
cv2.putText(img, 'OpenCV Features', (200, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

print("测试图像已创建")

# Harris角点检测
print("\n=== Harris角点检测 ===")

# 转换为灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Harris角点检测
# blockSize: 邻域大小
# ksize: Sobel算子孔径参数
# k: Harris角点检测器参数
corners = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)

# 标记角点
img_harris = img.copy()
img_harris[corners > 0.01 * corners.max()] = [0, 0, 255]

print(f"检测到的角点数量: {np.sum(corners > 0.01 * corners.max())}")

# Shi-Tomasi角点检测
print("\n=== Shi-Tomasi角点检测 ===")

# 设置参数
maxCorners = 100
qualityLevel = 0.01
minDistance = 10

# Shi-Tomasi角点检测
corners_st = cv2.goodFeaturesToTrack(gray, maxCorners, qualityLevel, minDistance)

print(f"检测到的Shi-Tomasi角点数量: {len(corners_st)}")

# 绘制Shi-Tomasi角点
img_st = img.copy()
if corners_st is not None:
    for corner in corners_st:
        x, y = corner.ravel()
        cv2.circle(img_st, (int(x), int(y)), 5, (0, 255, 0), 2)

# SIFT特征检测
print("\n=== SIFT特征检测 ===")

# 创建SIFT对象
sift = cv2.SIFT_Create()

# 检测关键点和描述符
keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)

print(f"SIFT检测到的关键点数量: {len(keypoints_sift)}")

# 绘制SIFT关键点
img_sift = cv2.drawKeypoints(img, keypoints_sift, None, 
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# ORB特征检测
print("\n=== ORB特征检测 ===")

# 创建ORB对象
orb = cv2.ORB_create(nfeatures=500)

# 检测关键点和描述符
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)

print(f"ORB检测到的关键点数量: {len(keypoints_orb)}")

# 绘制ORB关键点
img_orb = cv2.drawKeypoints(img, keypoints_orb, None, color=(0, 255, 0))

# AKAZE特征检测
print("\n=== AKAZE特征检测 ===")

# 创建AKAZE对象
akaze = cv2.AKAZE_create()

# 检测关键点和描述符
keypoints_akaze, descriptors_akaze = akaze.detectAndCompute(gray, None)

print(f"AKAZE检测到的关键点数量: {len(keypoints_akaze)}")

# 绘制AKAZE关键点
img_akaze = cv2.drawKeypoints(img, keypoints_akaze, None, color=(0, 255, 0))

# 可视化特征检测结果
print("\n=== 可视化特征检测结果 ===")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title('Harris角点检测')
axes[0, 1].axis('off')

axes[0, 2].imshow(cv2.cvtColor(img_st, cv2.COLOR_BGR2RGB))
axes[0, 2].set_title('Shi-Tomasi角点检测')
axes[0, 2].axis('off')

axes[1, 0].imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'SIFT特征 ({len(keypoints_sift)}个关键点)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f'ORB特征 ({len(keypoints_orb)}个关键点)')
axes[1, 1].axis('off')

axes[1, 2].imshow(cv2.cvtColor(img_akaze, cv2.COLOR_BGR2RGB))
axes[1, 2].set_title(f'AKAZE特征 ({len(keypoints_akaze)}个关键点)')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# 特征匹配
print("\n=== 特征匹配 ===")

# 创建另一个测试图像（稍微旋转和缩放）
img2 = img.copy()

# 旋转图像
(h, w) = img2.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 15, 0.9)
img2_rotated = cv2.warpAffine(img2, M, (w, h))

# 转换为灰度
gray2 = cv2.cvtColor(img2_rotated, cv2.COLOR_BGR2GRAY)

# 使用ORB进行特征匹配
print("使用ORB进行特征匹配...")

# 检测两个图像的特征
keypoints1, descriptors1 = orb.detectAndCompute(gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

# 创建匹配器
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配描述符
matches = bf.match(descriptors1, descriptors2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

print(f"匹配数量: {len(matches)}")

# 绘制匹配结果
img_matches = cv2.drawMatches(img, keypoints1, img2_rotated, keypoints2, 
                              matches[:50], None, flags=2)

plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('ORB特征匹配结果')
plt.axis('off')
plt.show()

# KNN匹配
print("\n=== KNN匹配 ===")

# 使用KNN匹配
knn_matches = bf.knnMatch(descriptors1, descriptors2, k=2)

# 应用比率测试
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

print(f"好的匹配数量（比率测试后）: {len(good_matches)}")

# 绘制KNN匹配结果
img_knn_matches = cv2.drawMatchesKnn(img, keypoints1, img2_rotated, keypoints2,
                                      [[m] for m in good_matches[:30]], None, flags=2)

plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_knn_matches, cv2.COLOR_BGR2RGB))
plt.title('KNN特征匹配结果')
plt.axis('off')
plt.show()

# 霍夫变换检测直线和圆
print("\n=== 霍夫变换 ===")

# 创建一个带有直线的图像
lines_img = np.zeros((400, 600), dtype=np.uint8)
cv2.line(lines_img, (50, 100), (550, 100), 255, 2)
cv2.line(lines_img, (50, 200), (550, 200), 255, 2)
cv2.line(lines_img, (50, 300), (550, 300), 255, 2)
cv2.line(lines_img, (100, 50), (100, 350), 255, 2)
cv2.line(lines_img, (500, 50), (500, 350), 255, 2)

# 霍夫直线变换
lines = cv2.HoughLinesP(lines_img, rho=1, theta=np.pi/180, threshold=50,
                          minLineLength=50, maxLineGap=10)

print(f"检测到的直线段数量: {len(lines)}")

# 创建圆形图像
circles_img = np.zeros((400, 600), dtype=np.uint8)
cv2.circle(circles_img, (150, 200), 80, 255, 2)
cv2.circle(circles_img, (350, 150), 60, 255, 2)
cv2.circle(circles_img, (450, 280), 50, 255, 2)

# 霍夫圆变换
circles = cv2.HoughCircles(circles_img, cv2.HOUGH_GRADIENT, dp=1, 
                            minDist=30, param1=50, param2=30,
                            minRadius=30, maxRadius=100)

if circles is not None:
    circles = np.uint16(np.around(circles))
    print(f"检测到的圆数量: {len(circles[0])}")

# 模板匹配
print("\n=== 模板匹配 ===")

# 创建模板
template = img[60:90, 60:100]
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 在原图中查找模板
result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)

# 获取最佳匹配位置
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(f"最佳匹配位置: {max_loc}")
print(f"匹配分数: {max_val:.4f}")

# 绘制匹配结果
h, w = template_gray.shape
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
img_template_match = img.copy()
cv2.rectangle(img_template_match, top_left, bottom_right, (0, 255, 0), 2)

plt.figure(figsize=(8, 6))
plt.imshow(cv2.cvtColor(img_template_match, cv2.COLOR_BGR2RGB))
plt.title('模板匹配结果')
plt.axis('off')
plt.show()

print("\n特征检测与匹配学习完成！")