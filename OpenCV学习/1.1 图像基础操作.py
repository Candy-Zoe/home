# OpenCV图像基础操作学习
# 主要内容：图像读取、显示、保存、颜色空间转换

# 导入OpenCV库，通常使用cv2作为别名
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 创建一个简单的测试图像
print("=== 创建测试图像 ===")

# 创建一个300x300的RGB图像（全黑）
img = np.zeros((300, 300, 3), dtype=np.uint8)

# 在图像上绘制一个红色的矩形
cv2.rectangle(img, (50, 50), (250, 250), (0, 0, 255), 3)

# 在图像中心绘制一个绿色的圆
cv2.circle(img, (150, 150), 50, (0, 255, 0), -1)

# 在图像上添加文字
cv2.putText(img, 'OpenCV', (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

print("测试图像已创建")

# 图像基本属性
print("\n=== 图像属性 ===")
print(f"图像形状: {img.shape}")  # (高度, 宽度, 通道数)
print(f"图像高度: {img.shape[0]} 像素")
print(f"图像宽度: {img.shape[1]} 像素")
print(f"通道数: {img.shape[2]}")
print(f"数据类型: {img.dtype}")
print(f"图像大小: {img.size} 字节")

# 显示图像
print("\n=== 显示图像 ===")

# 使用matplotlib显示图像（注意颜色空间转换）
plt.figure(figsize=(8, 6))

# OpenCV读取的图像是BGR格式，需要转换为RGB格式显示
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.title('测试图像')
plt.axis('off')  # 关闭坐标轴
plt.show()

# 颜色空间转换
print("\n=== 颜色空间转换 ===")

# BGR转灰度
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(f"灰度图像形状: {gray.shape}")

# BGR转HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# BGR转YUV
yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

# 显示不同颜色空间的图像
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('灰度图')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(hsv)
plt.title('HSV')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(yuv)
plt.title('YUV')
plt.axis('off')

plt.tight_layout()
plt.show()

# 图像通道分离与合并
print("\n=== 通道分离与合并 ===")

# 分离BGR通道
b, g, r = cv2.split(img)

# 创建只保留红色通道的图像
img_red = img.copy()
img_red[:, :, 0] = 0  # B通道设为0
img_red[:, :, 1] = 0  # G通道设为0

# 创建只保留绿色通道的图像
img_green = img.copy()
img_green[:, :, 0] = 0
img_green[:, :, 2] = 0

# 创建只保留蓝色通道的图像
img_blue = img.copy()
img_blue[:, :, 1] = 0
img_blue[:, :, 2] = 0

# 显示分离后的通道
plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title('原始')
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(cv2.cvtColor(img_red, cv2.COLOR_BGR2RGB))
plt.title('红色通道')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(cv2.cvtColor(img_green, cv2.COLOR_BGR2RGB))
plt.title('绿色通道')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(cv2.cvtColor(img_blue, cv2.COLOR_BGR2RGB))
plt.title('蓝色通道')
plt.axis('off')

plt.tight_layout()
plt.show()

# 保存图像
print("\n=== 保存图像 ===")

# 保存为PNG格式
cv2.imwrite('test_image.png', img)
print("图像已保存为 test_image.png")

# 保存为JPEG格式
cv2.imwrite('test_image.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
print("图像已保存为 test_image.jpg")

# 清理测试文件
import os
for f in ['test_image.png', 'test_image.jpg']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除 {f}")