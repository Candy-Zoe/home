# OpenCV特征匹配与图像检索学习
# 主要内容：SIFT/SURF/ORB特征提取、特征匹配、FLANN匹配、图像检索

# 导入必要的库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
print("=== 读取图像 ===")

# 创建示例图像
# 创建一个带有简单特征的图像
img1 = np.zeros((200, 200, 3), dtype=np.uint8)
# 添加一些特征点
cv2.circle(img1, (50, 50), 20, (255, 0, 0), -1)
cv2.rectangle(img1, (120, 30), (180, 90), (0, 255, 0), -1)
cv2.line(img1, (30, 120), (170, 180), (0, 0, 255), 3)
cv2.putText(img1, 'Test', (60, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

# 创建一个相似但有变化的图像
img2 = np.zeros((200, 200, 3), dtype=np.uint8)
cv2.circle(img2, (55, 55), 22, (255, 0, 0), -1)  # 稍微移动并放大
cv2.rectangle(img2, (115, 35), (175, 95), (0, 200, 0), -1)  # 稍微移动并改变颜色
cv2.line(img2, (35, 125), (175, 185), (0, 0, 200), 2)  # 改变线宽和位置
cv2.putText(img2, 'Test', (55, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (200, 200, 0), 2)

# 也可以读取真实图像
# img1 = cv2.imread('image1.jpg')
# img2 = cv2.imread('image2.jpg')

print(f"图像1形状: {img1.shape}")
print(f"图像2形状: {img2.shape}")

# 显示图像
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title('图像1')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[1].set_title('图像2')
axes[1].axis('off')

plt.tight_layout()
plt.show()

# SIFT特征提取
print("\n=== SIFT特征提取 ===")

# 创建SIFT检测器
sift = cv2.SIFT_create()

# 检测关键点和描述符
keypoints1, descriptors1 = sift.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
keypoints2, descriptors2 = sift.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

print(f"图像1检测到的关键点数量: {len(keypoints1)}")
print(f"图像2检测到的关键点数量: {len(keypoints2)}")
print(f"描述符维度: {descriptors1.shape}")

# 绘制关键点
img1_kp = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kp = cv2.drawKeypoints(img2, keypoints2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
axes[0].set_title('图像1的SIFT关键点')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
axes[1].set_title('图像2的SIFT关键点')
axes[1].axis('off')

plt.suptitle('SIFT关键点检测')
plt.tight_layout()
plt.show()

# ORB特征提取
print("\n=== ORB特征提取 ===")

# 创建ORB检测器
orb = cv2.ORB_create(nfeatures=500)

# 检测关键点和描述符
keypoints_orb1, descriptors_orb1 = orb.detectAndCompute(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY), None)
keypoints_orb2, descriptors_orb2 = orb.detectAndCompute(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY), None)

print(f"图像1检测到的ORB关键点数量: {len(keypoints_orb1)}")
print(f"图像2检测到的ORB关键点数量: {len(keypoints_orb2)}")

# 绘制关键点
img1_orb = cv2.drawKeypoints(img1, keypoints_orb1, None, color=(0, 255, 0), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_orb = cv2.drawKeypoints(img2, keypoints_orb2, None, color=(0, 255, 0), 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(cv2.cvtColor(img1_orb, cv2.COLOR_BGR2RGB))
axes[0].set_title('图像1的ORB关键点')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img2_orb, cv2.COLOR_BGR2RGB))
axes[1].set_title('图像2的ORB关键点')
axes[1].axis('off')

plt.suptitle('ORB关键点检测')
plt.tight_layout()
plt.show()

# 暴力匹配 (Brute-force Matching)
print("\n=== 暴力匹配 ===")

# 创建BFMatcher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 匹配SIFT描述符
matches = bf.match(descriptors1, descriptors2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

print(f"匹配数量: {len(matches)}")
print(f"最佳匹配距离: {matches[0].distance:.2f}")
print(f"最差匹配距离: {matches[-1].distance:.2f}")

# 绘制匹配结果
img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, 
                              matches[:20], None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 可视化
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title('SIFT特征匹配结果 (前20个)')
plt.axis('off')
plt.show()

# KNN匹配
print("\n=== KNN匹配 ===")

# 使用KNN匹配
bf_knn = cv2.BFMatcher()
knn_matches = bf_knn.knnMatch(descriptors1, descriptors2, k=2)

# 应用比例测试 (Lowe's ratio test)
good_matches = []
for m, n in knn_matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

print(f"通过比例测试的匹配数量: {len(good_matches)}")

# 绘制KNN匹配结果
img_knn_matches = cv2.drawMatchesKnn(img1, keypoints1, img2, keypoints2, 
                                      good_matches, None, 
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 可视化
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_knn_matches, cv2.COLOR_BGR2RGB))
plt.title('KNN匹配结果 (通过比例测试)')
plt.axis('off')
plt.show()

# FLANN匹配
print("\n=== FLANN匹配 ===")

# FLANN参数
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

# 创建FLANN匹配器
flann = cv2.FlannBasedMatcher(index_params, search_params)

# 匹配描述符
flann_matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 应用比例测试
good_flann_matches = []
for m, n in flann_matches:
    if m.distance < 0.7 * n.distance:
        good_flann_matches.append(m)

print(f"FLANN匹配数量: {len(good_flann_matches)}")

# 绘制匹配结果
img_flann_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, 
                                     good_flann_matches[:20], None, 
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 可视化
plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_flann_matches, cv2.COLOR_BGR2RGB))
plt.title('FLANN特征匹配结果')
plt.axis('off')
plt.show()

# 单应性变换与图像拼接
print("\n=== 单应性变换 ===")

# 获取匹配点的坐标
src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_flann_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_flann_matches]).reshape(-1, 1, 2)

# 计算单应性矩阵
homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

print(f"单应性矩阵:\n{homography}")
print(f"RANSAC内点数量: {np.sum(mask)}")

# 应用单应性变换
h, w = img1.shape[:2]
img1_warped = cv2.warpPerspective(img1, homography, (w, h))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始图像1')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img1_warped, cv2.COLOR_BGR2RGB))
axes[1].set_title('变换后图像1')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
axes[2].set_title('图像2')
axes[2].axis('off')

plt.suptitle('单应性变换结果')
plt.tight_layout()
plt.show()

# 图像检索示例
print("\n=== 图像检索示例 ===")

# 创建图像数据库
database_images = []
database_descriptors = []

# 添加一些"数据库"图像（使用原始图像的变体）
for i in range(4):
    # 创建不同缩放和旋转的版本
    scale = 0.8 + i * 0.1
    angle = i * 15
    
    # 缩放
    h, w = img1.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    scaled = cv2.resize(img1, new_size)
    
    # 创建旋转矩阵
    center = (new_size[0] // 2, new_size[1] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # 旋转
    rotated = cv2.warpAffine(scaled, rotation_matrix, new_size)
    
    # 提取特征
    _, desc = sift.detectAndCompute(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), None)
    
    database_images.append(rotated)
    database_descriptors.append(desc)
    
    print(f"数据库图像 {i+1}: 尺寸={new_size}, 角度={angle}度, 特征数={len(_)}")

# 查询图像
query_img = img2
_, query_desc = sift.detectAndCompute(cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY), None)

# 计算与数据库中每个图像的匹配数
scores = []
for i, db_desc in enumerate(database_descriptors):
    if db_desc is not None and len(db_desc) > 0:
        matches = bf_knn.knnMatch(query_desc, db_desc, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)
        scores.append((i, len(good)))
        print(f"与数据库图像 {i+1} 的匹配数: {len(good)}")

# 找到最佳匹配
best_idx = max(scores, key=lambda x: x[1])[0]
print(f"\n最佳匹配: 数据库图像 {best_idx+1}")

# 可视化检索结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('查询图像')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(database_images[best_idx], cv2.COLOR_BGR2RGB))
axes[1].set_title(f'最佳匹配 (图像{best_idx+1})')
axes[1].axis('off')

axes[2].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
axes[2].set_title('原始图像')
axes[2].axis('off')

plt.suptitle('图像检索结果')
plt.tight_layout()
plt.show()

# 特征匹配质量评估
print("\n=== 特征匹配质量评估 ===")

# 计算匹配得分
matching_scores = []

# 使用不同的距离阈值
for threshold in [0.6, 0.7, 0.8]:
    good_count = sum(1 for m, n in knn_matches if m.distance < threshold * n.distance)
    matching_scores.append((threshold, good_count))
    print(f"阈值 {threshold}: 匹配数={good_count}")

# 可视化得分
thresholds, counts = zip(*matching_scores)
plt.figure(figsize=(8, 5))
plt.plot(thresholds, counts, marker='o', linestyle='-', color='b')
plt.xlabel('距离阈值')
plt.ylabel('匹配数量')
plt.title('不同阈值下的匹配数量')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 使用匹配点进行姿态估计
print("\n=== 使用匹配点进行姿态估计 ===")

# 假设已知相机内参（简化示例）
K = np.array([[1000, 0, 100],
              [0, 1000, 100],
              [0, 0, 1]], dtype=np.float32)

# 计算本质矩阵
try:
    essential_matrix, mask_e = cv2.findEssentialMat(src_pts, dst_pts, K, cv2.RANSAC, 0.999, 1.0)
    print(f"本质矩阵计算成功")
    
    # 从本质矩阵恢复姿态
    _, R, t, mask_pose = cv2.recoverPose(essential_matrix, src_pts, dst_pts, K)
    print(f"旋转矩阵:\n{R}")
    print(f"平移向量:\n{t}")
except Exception as e:
    print(f"姿态估计失败: {e}")

# 使用ORB进行实时匹配演示
print("\n=== ORB vs SIFT对比 ===")

# ORB匹配
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
orb_matches = bf_orb.match(descriptors_orb1, descriptors_orb2)
orb_matches = sorted(orb_matches, key=lambda x: x.distance)

# SIFT匹配
sift_matches = sorted(matches, key=lambda x: x.distance)

# 对比匹配结果
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# ORB匹配
img_orb_match = cv2.drawMatches(img1, keypoints_orb1, img2, keypoints_orb2, 
                                 orb_matches[:15], None, 
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
axes[0].imshow(cv2.cvtColor(img_orb_match, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'ORB匹配 (前15个)')
axes[0].axis('off')

# SIFT匹配
img_sift_match = cv2.drawMatches(img1, keypoints1, img2, keypoints2, 
                                  sift_matches[:15], None, 
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
axes[1].imshow(cv2.cvtColor(img_sift_match, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'SIFT匹配 (前15个)')
axes[1].axis('off')

plt.suptitle('ORB vs SIFT匹配对比')
plt.tight_layout()
plt.show()

print("\nOpenCV特征匹配与图像检索学习完成！")
