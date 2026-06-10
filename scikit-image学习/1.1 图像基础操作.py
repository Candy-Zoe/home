# scikit-image图像基础操作学习
# 主要内容：图像读取、显示、保存、颜色空间转换

from skimage import io, color, transform
import matplotlib.pyplot as plt

print("=== 读取图像 ===")
img = io.imread('example.jpg')
print(f"图像形状: {img.shape}")
print(f"图像类型: {img.dtype}")

print("\n=== 显示图像 ===")
plt.imshow(img)
plt.axis('off')
plt.title('原始图像')
plt.show()

print("\n=== 颜色空间转换 ===")
gray = color.rgb2gray(img)
hsv = color.rgb2hsv(img)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('灰度图像')

plt.subplot(1, 2, 2)
plt.imshow(hsv)
plt.title('HSV图像')
plt.show()

print("\n=== 图像缩放 ===")
resized = transform.resize(img, (200, 300))
print(f"原始大小: {img.shape}")
print(f"缩放后大小: {resized.shape}")

print("\n=== 图像旋转 ===")
rotated = transform.rotate(img, 45)
plt.imshow(rotated)
plt.title('旋转45度')
plt.show()

print("\n=== 保存图像 ===")
io.imsave('gray_skimage.jpg', (gray * 255).astype('uint8'))
print("灰度图像已保存")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('gray_skimage.jpg'):
    os.remove('gray_skimage.jpg')
    print("已删除测试文件")