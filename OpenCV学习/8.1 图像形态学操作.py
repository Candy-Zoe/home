# OpenCV图像形态学操作学习
# 主要内容：腐蚀、膨胀、开运算、闭运算、顶帽运算、黑帽运算、形态学梯度

# 导入必要的库
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
print("=== 读取图像 ===")

# 读取灰度图像
img = cv2.imread('test_image.jpg', cv2.IMREAD_GRAYSCALE)

# 如果没有读取到图像，创建一个示例图像
if img is None:
    print("未找到测试图像，创建示例图像")
    # 创建一个带有噪声的二值图像
    img = np.zeros((200, 200), dtype=np.uint8)
    # 添加一些物体
    img[50:100, 50:100] = 255  # 正方形
    img[120:160, 80:120] = 255  # 矩形
    img[70:90, 140:160] = 255   # 小矩形
    # 添加噪声
    noise = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
    img = cv2.bitwise_or(img, noise & 0x30)

print(f"图像形状: {img.shape}")

# 显示原始图像
plt.figure(figsize=(10, 6))
plt.imshow(img, cmap='gray')
plt.title('原始图像')
plt.axis('off')
plt.show()

# 定义结构元素（核）
print("\n=== 定义结构元素 ===")

# 创建不同形状的结构元素
kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

print(f"正方形结构元素:\n{kernel_square}")
print(f"\n圆形结构元素:\n{kernel_circle}")
print(f"\n十字结构元素:\n{kernel_cross}")

# 腐蚀操作
print("\n=== 腐蚀操作 (Erosion) ===")

# 使用不同大小的核进行腐蚀
erosion_3x3 = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
erosion_5x5 = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
erosion_7x7 = cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(erosion_3x3, cmap='gray')
axes[0, 1].set_title('腐蚀 3x3')
axes[0, 1].axis('off')

axes[1, 0].imshow(erosion_5x5, cmap='gray')
axes[1, 0].set_title('腐蚀 5x5')
axes[1, 0].axis('off')

axes[1, 1].imshow(erosion_7x7, cmap='gray')
axes[1, 1].set_title('腐蚀 7x7')
axes[1, 1].axis('off')

plt.suptitle('腐蚀操作对比')
plt.tight_layout()
plt.show()

# 膨胀操作
print("\n=== 膨胀操作 (Dilation) ===")

# 使用不同大小的核进行膨胀
dilation_3x3 = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
dilation_5x5 = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
dilation_7x7 = cv2.dilate(img, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(dilation_3x3, cmap='gray')
axes[0, 1].set_title('膨胀 3x3')
axes[0, 1].axis('off')

axes[1, 0].imshow(dilation_5x5, cmap='gray')
axes[1, 0].set_title('膨胀 5x5')
axes[1, 0].axis('off')

axes[1, 1].imshow(dilation_7x7, cmap='gray')
axes[1, 1].set_title('膨胀 7x7')
axes[1, 1].axis('off')

plt.suptitle('膨胀操作对比')
plt.tight_layout()
plt.show()

# 开运算 (先腐蚀后膨胀)
print("\n=== 开运算 (Opening) ===")

# 开运算 = 腐蚀 + 膨胀
opening_3x3 = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
opening_5x5 = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(opening_3x3, cmap='gray')
axes[1].set_title('开运算 3x3')
axes[1].axis('off')

axes[2].imshow(opening_5x5, cmap='gray')
axes[2].set_title('开运算 5x5')
axes[2].axis('off')

plt.suptitle('开运算对比')
plt.tight_layout()
plt.show()

# 闭运算 (先膨胀后腐蚀)
print("\n=== 闭运算 (Closing) ===")

# 闭运算 = 膨胀 + 腐蚀
closing_3x3 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
closing_5x5 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(closing_3x3, cmap='gray')
axes[1].set_title('闭运算 3x3')
axes[1].axis('off')

axes[2].imshow(closing_5x5, cmap='gray')
axes[2].set_title('闭运算 5x5')
axes[2].axis('off')

plt.suptitle('闭运算对比')
plt.tight_layout()
plt.show()

# 形态学梯度
print("\n=== 形态学梯度 (Gradient) ===")

# 形态学梯度 = 膨胀 - 腐蚀
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# 内部梯度和外部梯度
gradient_inner = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
gradient_outer = img - cv2.erode(img, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(gradient, cmap='gray')
axes[1].set_title('形态学梯度')
axes[1].axis('off')

axes[2].imshow(gradient_outer, cmap='gray')
axes[2].set_title('外部梯度')
axes[2].axis('off')

plt.suptitle('形态学梯度对比')
plt.tight_layout()
plt.show()

# 顶帽运算 (Top Hat)
print("\n=== 顶帽运算 (Top Hat) ===")

# 顶帽 = 原始图像 - 开运算结果
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(tophat, cmap='gray')
axes[1].set_title('顶帽运算结果')
axes[1].axis('off')

plt.suptitle('顶帽运算')
plt.tight_layout()
plt.show()

# 黑帽运算 (Black Hat)
print("\n=== 黑帽运算 (Black Hat) ===")

# 黑帽 = 闭运算结果 - 原始图像
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(img, cmap='gray')
axes[0].set_title('原始图像')
axes[0].axis('off')

axes[1].imshow(blackhat, cmap='gray')
axes[1].set_title('黑帽运算结果')
axes[1].axis('off')

plt.suptitle('黑帽运算')
plt.tight_layout()
plt.show()

# 实际应用示例 - 去除噪声
print("\n=== 实际应用示例 ===")

# 创建带有椒盐噪声的图像
noisy_img = img.copy()
# 添加椒盐噪声
salt_pepper = np.random.randint(0, 256, img.shape)
noisy_img[salt_pepper < 10] = 0   # 胡椒噪声
noisy_img[salt_pepper > 245] = 255  # 盐噪声

print("\n1. 使用开运算去除盐噪声")
opening_result = cv2.morphologyEx(noisy_img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

print("\n2. 使用闭运算去除胡椒噪声")
closing_result = cv2.morphologyEx(opening_result, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(noisy_img, cmap='gray')
axes[0, 1].set_title('添加噪声后')
axes[0, 1].axis('off')

axes[1, 0].imshow(opening_result, cmap='gray')
axes[1, 0].set_title('开运算去噪')
axes[1, 0].axis('off')

axes[1, 1].imshow(closing_result, cmap='gray')
axes[1, 1].set_title('闭运算去噪')
axes[1, 1].axis('off')

plt.suptitle('形态学操作去除噪声')
plt.tight_layout()
plt.show()

# 实际应用示例 - 文本提取
print("\n3. 使用形态学操作提取文本")

# 创建文本图像
text_img = np.zeros((100, 300), dtype=np.uint8)
cv2.putText(text_img, 'Hello OpenCV!', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
# 添加噪声
text_noisy = text_img.copy()
text_noisy[np.random.randint(0, 100, 50), np.random.randint(0, 300, 50)] = 255

# 使用开运算去除小噪声点
text_clean = cv2.morphologyEx(text_noisy, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(text_img, cmap='gray')
axes[0].set_title('原始文本')
axes[0].axis('off')

axes[1].imshow(text_noisy, cmap='gray')
axes[1].set_title('添加噪声')
axes[1].axis('off')

axes[2].imshow(text_clean, cmap='gray')
axes[2].set_title('开运算处理')
axes[2].axis('off')

plt.suptitle('文本提取示例')
plt.tight_layout()
plt.show()

# 实际应用示例 - 边界提取
print("\n4. 使用形态学梯度提取边界")

# 创建一个简单的形状
shape_img = np.zeros((200, 200), dtype=np.uint8)
cv2.rectangle(shape_img, (50, 50), (150, 150), 255, 3)  # 矩形边框
cv2.circle(shape_img, (100, 100), 30, 255, 2)            # 圆形

# 使用梯度提取边界
gradient_shape = cv2.morphologyEx(shape_img, cv2.MORPH_GRADIENT, 
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(shape_img, cmap='gray')
axes[0].set_title('原始形状')
axes[0].axis('off')

axes[1].imshow(gradient_shape, cmap='gray')
axes[1].set_title('边界提取')
axes[1].axis('off')

plt.suptitle('边界提取示例')
plt.tight_layout()
plt.show()

# 不同结构元素的效果对比
print("\n=== 不同结构元素的效果对比 ===")

# 创建测试图像
test_img = np.zeros((150, 150), dtype=np.uint8)
# 添加不同形状的物体
cv2.rectangle(test_img, (20, 20), (60, 60), 255, -1)    # 正方形
cv2.circle(test_img, (100, 40), 20, 255, -1)            # 圆形
cv2.ellipse(test_img, (50, 100), (25, 15), 0, 0, 360, 255, -1)  # 椭圆
cv2.line(test_img, (100, 80), (140, 140), 255, 3)       # 线段

# 使用不同结构元素进行开运算
kernel_square = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
kernel_cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

opening_square = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, kernel_square)
opening_circle = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, kernel_circle)
opening_cross = cv2.morphologyEx(test_img, cv2.MORPH_OPEN, kernel_cross)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(test_img, cmap='gray')
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(opening_square, cmap='gray')
axes[0, 1].set_title('正方形结构元素')
axes[0, 1].axis('off')

axes[1, 0].imshow(opening_circle, cmap='gray')
axes[1, 0].set_title('圆形结构元素')
axes[1, 0].axis('off')

axes[1, 1].imshow(opening_cross, cmap='gray')
axes[1, 1].set_title('十字结构元素')
axes[1, 1].axis('off')

plt.suptitle('不同结构元素的开运算效果')
plt.tight_layout()
plt.show()

# 使用inRange和形态学操作进行颜色分割
print("\n=== 颜色分割与形态学操作 ===")

# 创建彩色图像
color_img = np.zeros((200, 200, 3), dtype=np.uint8)
# 添加不同颜色的区域
color_img[20:80, 20:80] = [0, 255, 0]      # 绿色
color_img[120:180, 20:80] = [0, 0, 255]    # 红色
color_img[20:80, 120:180] = [255, 0, 0]    # 蓝色
color_img[120:180, 120:180] = [0, 255, 255] # 青色

# 转换为HSV颜色空间
hsv_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)

# 定义绿色范围
lower_green = np.array([40, 40, 40])
upper_green = np.array([80, 255, 255])

# 创建掩膜
mask = cv2.inRange(hsv_img, lower_green, upper_green)

# 使用形态学操作去除噪声
mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))
axes[0].set_title('原始彩色图像')
axes[0].axis('off')

axes[1].imshow(mask, cmap='gray')
axes[1].set_title('颜色掩膜')
axes[1].axis('off')

axes[2].imshow(mask_clean, cmap='gray')
axes[2].set_title('形态学处理后')
axes[2].axis('off')

plt.suptitle('颜色分割与形态学处理')
plt.tight_layout()
plt.show()

print("\nOpenCV图像形态学操作学习完成！")
