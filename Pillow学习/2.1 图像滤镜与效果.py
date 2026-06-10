# Pillow图像滤镜与效果学习
# 主要内容：内置滤镜、自定义效果、图像增强

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

print("=== 内置滤镜 ===")
img = Image.open('example.jpg')

blur = img.filter(ImageFilter.BLUR)
plt.imshow(blur)
plt.title('模糊效果')
plt.show()

sharp = img.filter(ImageFilter.SHARPEN)
plt.imshow(sharp)
plt.title('锐化效果')
plt.show()

edge = img.filter(ImageFilter.FIND_EDGES)
plt.imshow(edge, cmap='gray')
plt.title('边缘检测')
plt.show()

print("\n=== 颜色调整 ===")
from PIL import ImageEnhance

enhancer = ImageEnhance.Brightness(img)
bright_img = enhancer.enhance(1.5)
plt.imshow(bright_img)
plt.title('亮度增强')
plt.show()

enhancer = ImageEnhance.Contrast(img)
contrast_img = enhancer.enhance(1.5)
plt.imshow(contrast_img)
plt.title('对比度增强')
plt.show()

enhancer = ImageEnhance.Color(img)
color_img = enhancer.enhance(2.0)
plt.imshow(color_img)
plt.title('饱和度增强')
plt.show()

print("\n=== 像素操作 ===")
pixels = img.load()
width, height = img.size

for i in range(width):
    for j in range(height):
        r, g, b = pixels[i, j]
        pixels[i, j] = (r, 0, b)

plt.imshow(img)
plt.title('绿色通道置零')
plt.show()