# Pillow图像滤镜进阶学习
# 主要内容：自定义滤镜、卷积核、边缘检测、图像增强

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建测试图像 ===")
img = Image.new('RGB', (200, 200), color=(255, 255, 255))
for i in range(0, 200, 20):
    for j in range(0, 200, 20):
        img.putpixel((i, j), (0, 0, 0))

plt.imshow(img)
plt.title('原始图像')
plt.show()

print("\n=== 内置滤镜 ===")
blur = img.filter(ImageFilter.BLUR)
contour = img.filter(ImageFilter.CONTOUR)
detail = img.filter(ImageFilter.DETAIL)
edge_enhance = img.filter(ImageFilter.EDGE_ENHANCE)
emboss = img.filter(ImageFilter.EMBOSS)

plt.figure(figsize=(15, 5))
plt.subplot(1, 5, 1)
plt.imshow(blur)
plt.title('BLUR')

plt.subplot(1, 5, 2)
plt.imshow(contour)
plt.title('CONTOUR')

plt.subplot(1, 5, 3)
plt.imshow(detail)
plt.title('DETAIL')

plt.subplot(1, 5, 4)
plt.imshow(edge_enhance)
plt.title('EDGE_ENHANCE')

plt.subplot(1, 5, 5)
plt.imshow(emboss)
plt.title('EMBOSS')
plt.show()

print("\n=== 自定义卷积核 ===")
kernel_sharpen = (
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
)
sharpened = img.filter(ImageFilter.Kernel((3, 3), kernel_sharpen))

kernel_blur = (
    1/9, 1/9, 1/9,
    1/9, 1/9, 1/9,
    1/9, 1/9, 1/9
)
custom_blur = img.filter(ImageFilter.Kernel((3, 3), kernel_blur))

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sharpened)
plt.title('锐化')

plt.subplot(1, 2, 2)
plt.imshow(custom_blur)
plt.title('自定义模糊')
plt.show()

print("\n=== 边缘检测 ===")
kernel_edge = (
    -1, -1, -1,
    -1, 8, -1,
    -1, -1, -1
)
edges = img.filter(ImageFilter.Kernel((3, 3), kernel_edge))

plt.imshow(edges)
plt.title('边缘检测')
plt.show()

print("\n=== 高斯模糊 ===")
gaussian_blur = img.filter(ImageFilter.GaussianBlur(radius=2))
plt.imshow(gaussian_blur)
plt.title('高斯模糊')
plt.show()

print("\n=== 图像增强 ===")
from PIL import ImageEnhance

enhancer = ImageEnhance.Contrast(img)
high_contrast = enhancer.enhance(2.0)

enhancer = ImageEnhance.Brightness(img)
bright = enhancer.enhance(1.5)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(high_contrast)
plt.title('高对比度')

plt.subplot(1, 2, 2)
plt.imshow(bright)
plt.title('高亮')
plt.show()