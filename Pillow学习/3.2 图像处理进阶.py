# Pillow图像识别与处理进阶学习
# 主要内容：OCR文字识别、条形码检测、图像增强、批量处理

from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建测试图像 ===")
img = Image.new('RGB', (400, 200), color=(255, 255, 255))
draw = ImageDraw.Draw(img)

try:
    font = ImageFont.truetype("arial.ttf", 30)
except:
    font = ImageFont.load_default()

draw.text((50, 80), "Hello OCR", fill=(0, 0, 0), font=font)

plt.imshow(img)
plt.title('测试图像')
plt.show()

print("\n=== 图像增强 ===")
enhancer = ImageEnhance.Sharpness(img)
sharpened = enhancer.enhance(2.0)

enhancer = ImageEnhance.Color(img)
color_enhanced = enhancer.enhance(1.5)

enhancer = ImageEnhance.Contrast(img)
contrasted = enhancer.enhance(1.5)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sharpened)
plt.title('锐化')

plt.subplot(1, 3, 2)
plt.imshow(color_enhanced)
plt.title('色彩增强')

plt.subplot(1, 3, 3)
plt.imshow(contrasted)
plt.title('对比度增强')
plt.show()

print("\n=== 图像混合 ===")
img1 = Image.new('RGB', (200, 200), color=(255, 0, 0))
img2 = Image.new('RGB', (200, 200), color=(0, 0, 255))

blended = Image.blend(img1, img2, alpha=0.5)

plt.imshow(blended)
plt.title('混合图像')
plt.show()

print("\n=== 图像合成 ===")
background = Image.new('RGB', (400, 300), color=(255, 255, 255))
foreground = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))

background.paste(foreground, (150, 100), foreground)

plt.imshow(background)
plt.title('合成图像')
plt.show()

print("\n=== 图像变换矩阵 ===")
from PIL import ImageTransform

img = Image.new('RGB', (200, 200), color=(100, 150, 200))

transform = ImageTransform.AffineTransform(
    matrix=(1, 0.2, 0, 0.2, 1, 0)
)

transformed = img.transform((200, 200), Image.AFFINE, transform.matrix)

plt.imshow(transformed)
plt.title('仿射变换')
plt.show()

print("\n=== 图像卷积 ===")
kernel = [
    0, -1, 0,
    -1, 5, -1,
    0, -1, 0
]

convolved = img.filter(ImageFilter.Kernel((3, 3), kernel))

plt.imshow(convolved)
plt.title('卷积结果')
plt.show()

print("\n=== 批量处理 ===")
images = []
for i in range(5):
    img = Image.new('RGB', (100, 100), color=(i*50, i*50, i*50))
    images.append(img)

print(f"处理图像数量: {len(images)}")

print("\n=== 图像模式转换 ===")
img_gray = img.convert('L')
img_rgb = img.convert('RGB')
img_rgba = img.convert('RGBA')

print(f"灰度模式: {img_gray.mode}")
print(f"RGB模式: {img_rgb.mode}")
print(f"RGBA模式: {img_rgba.mode}")

print("\n=== 图像统计 ===")
img_array = np.array(img)
print(f"最小值: {img_array.min()}")
print(f"最大值: {img_array.max()}")
print(f"均值: {img_array.mean():.2f}")
print(f"标准差: {img_array.std():.2f}")

print("\n=== 图像直方图 ===")
gray = img.convert('L')
histogram = gray.histogram()

plt.figure(figsize=(10, 5))
plt.bar(range(len(histogram)), histogram)
plt.title('灰度直方图')
plt.xlabel('灰度值')
plt.ylabel('像素数量')
plt.show()

print("\n=== 图像通道分离 ===")
r, g, b = img.split()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(r, cmap='Reds')
plt.title('红通道')

plt.subplot(1, 3, 2)
plt.imshow(g, cmap='Greens')
plt.title('绿通道')

plt.subplot(1, 3, 3)
plt.imshow(b, cmap='Blues')
plt.title('蓝通道')
plt.show()

print("\n=== 创建GIF ===")
frames = []
for i in range(10):
    frame = Image.new('RGB', (100, 100), color=(i*25, 100, 200-i*20))
    frames.append(frame)

frames[0].save(
    'animation.gif',
    save_all=True,
    append_images=frames[1:],
    duration=100,
    loop=0
)
print("GIF动画已创建")

print("\n=== 清理 ===")
import os
if os.path.exists('animation.gif'):
    os.remove('animation.gif')
    print("GIF文件已删除")