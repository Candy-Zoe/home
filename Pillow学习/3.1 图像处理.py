# Pillow图像处理高级学习
# 主要内容：图像滤镜、图像增强、图像变换、图像合成、图像滤镜

# 导入必要的库
from PIL import Image, ImageFilter, ImageEnhance, ImageChops, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

# Pillow基础
print("=== Pillow基础 ===")
print(f"Pillow版本信息: {Image.__version__}")

# 创建示例图像
print("\n=== 创建示例图像 ===")

# 创建一个彩色图像
width, height = 400, 300
img = Image.new('RGB', (width, height), color=(100, 150, 200))
print(f"创建图像: {width}x{height}, 模式: RGB")

# 在图像上绘制一些内容
draw = ImageDraw.Draw(img)
# 绘制矩形
draw.rectangle([50, 50, 150, 150], fill=(255, 100, 100), outline=(0, 0, 0), width=2)
# 绘制椭圆
draw.ellipse([200, 50, 350, 150], fill=(100, 255, 100), outline=(0, 0, 0), width=2)
# 绘制多边形
draw.polygon([(100, 250), (200, 150), (300, 250)], fill=(255, 255, 100), outline=(0, 0, 0))
# 绘制文字
draw.text((150, 130), "Pillow", fill=(0, 0, 0))

# 保存示例图像
img.save('sample_image.jpg')
print("示例图像已保存为 sample_image.jpg")

# 显示图像
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.title('原始图像')
plt.axis('off')
plt.show()

# 图像滤镜
print("\n=== 图像滤镜 ===")

# 创建不同的滤镜效果
filters = {
    'BLUR': ImageFilter.BLUR,
    'CONTOUR': ImageFilter.CONTOUR,
    'DETAIL': ImageFilter.DETAIL,
    'EDGE_ENHANCE': ImageFilter.EDGE_ENHANCE,
    'EDGE_ENHANCE_MORE': ImageFilter.EDGE_ENHANCE_MORE,
    'EMBOSS': ImageFilter.EMBOSS,
    'FIND_EDGES': ImageFilter.FIND_EDGES,
    'SHARPEN': ImageFilter.SHARPEN,
    'SMOOTH': ImageFilter.SMOOTH,
    'SMOOTH_MORE': ImageFilter.SMOOTH_MORE,
}

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for idx, (name, filter_type) in enumerate(filters.items()):
    filtered_img = img.filter(filter_type)
    axes[idx].imshow(filtered_img)
    axes[idx].set_title(name)
    axes[idx].axis('off')

plt.suptitle('Pillow图像滤镜效果', fontsize=14)
plt.tight_layout()
plt.show()

# 自定义滤镜
print("\n=== 自定义滤镜 ===")

# 创建自定义卷积核
class CustomFilter(ImageFilter.Filter):
    name = "Custom Filter"
    
    def __init__(self):
        # 定义3x3卷积核
        self.kernel = [
            -1, -1, 0,
            -1, 0, 1,
            0, 1, 1
        ]
    
    def filter(self, image):
        return image.filter(ImageFilter.Kernel((3, 3), self.kernel, scale=None))

custom_filtered = img.filter(CustomFilter())
plt.figure(figsize=(10, 6))
plt.imshow(custom_filtered)
plt.title('自定义滤镜')
plt.axis('off')
plt.show()

# 图像增强
print("\n=== 图像增强 ===")

# 亮度增强
enhancer = ImageEnhance.Brightness(img)
bright_images = [enhancer.enhance(factor) for factor in [0.3, 0.7, 1.0, 1.5, 2.0]]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for idx, (factor, bright_img) in enumerate(zip([0.3, 0.7, 1.0, 1.5, 2.0], bright_images)):
    axes[idx].imshow(bright_img)
    axes[idx].set_title(f'亮度: {factor}x')
    axes[idx].axis('off')

plt.suptitle('亮度增强效果', fontsize=14)
plt.tight_layout()
plt.show()

# 对比度增强
enhancer = ImageEnhance.Contrast(img)
contrast_images = [enhancer.enhance(factor) for factor in [0.5, 0.75, 1.0, 1.5, 2.0]]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for idx, (factor, contrast_img) in enumerate(zip([0.5, 0.75, 1.0, 1.5, 2.0], contrast_images)):
    axes[idx].imshow(contrast_img)
    axes[idx].set_title(f'对比度: {factor}x')
    axes[idx].axis('off')

plt.suptitle('对比度增强效果', fontsize=14)
plt.tight_layout()
plt.show()

# 色彩饱和度增强
enhancer = ImageEnhance.Color(img)
saturation_images = [enhancer.enhance(factor) for factor in [0.0, 0.5, 1.0, 1.5, 2.0]]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for idx, (factor, sat_img) in enumerate(zip([0.0, 0.5, 1.0, 1.5, 2.0], saturation_images)):
    axes[idx].imshow(sat_img)
    axes[idx].set_title(f'饱和度: {factor}x')
    axes[idx].axis('off')

plt.suptitle('饱和度增强效果', fontsize=14)
plt.tight_layout()
plt.show()

# 锐度增强
enhancer = ImageEnhance.Sharpness(img)
sharpness_images = [enhancer.enhance(factor) for factor in [0.0, 0.5, 1.0, 1.5, 3.0]]

fig, axes = plt.subplots(1, 5, figsize=(15, 4))
for idx, (factor, sharp_img) in enumerate(zip([0.0, 0.5, 1.0, 1.5, 3.0], sharpness_images)):
    axes[idx].imshow(sharp_img)
    axes[idx].set_title(f'锐度: {factor}x')
    axes[idx].axis('off')

plt.suptitle('锐度增强效果', fontsize=14)
plt.tight_layout()
plt.show()

# 图像变换
print("\n=== 图像变换 ===")

# 调整大小
print("1. 调整大小:")
resized = img.resize((200, 150))
print(f"   原始尺寸: {img.size}, 调整后: {resized.size}")

# 缩放
print("2. 缩放 (thumbnail):")
thumbnail = img.copy()
thumbnail.thumbnail((100, 100))
print(f"   缩放后: {thumbnail.size}")

# 旋转
print("3. 旋转:")
rotated = img.rotate(45, expand=True)
print(f"   旋转45度后尺寸: {rotated.size}")

# 翻转
print("4. 翻转:")
flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
print(f"   左右翻转: {flipped_h.size}")
print(f"   上下翻转: {flipped_v.size}")

# 裁剪
print("5. 裁剪:")
cropped = img.crop((100, 75, 300, 225))
print(f"   裁剪区域: (100, 75, 300, 225), 裁剪后: {cropped.size}")

# 可视化变换效果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(img)
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(resized)
axes[0, 1].set_title(f'调整大小 ({resized.size})')
axes[0, 1].axis('off')

axes[0, 2].imshow(rotated)
axes[0, 2].set_title('旋转45度')
axes[0, 2].axis('off')

axes[1, 0].imshow(flipped_h)
axes[1, 0].set_title('左右翻转')
axes[1, 0].axis('off')

axes[1, 1].imshow(flipped_v)
axes[1, 1].set_title('上下翻转')
axes[1, 1].axis('off')

axes[1, 2].imshow(cropped)
axes[1, 2].set_title(f'裁剪 ({cropped.size})')
axes[1, 2].axis('off')

plt.suptitle('图像变换效果', fontsize=14)
plt.tight_layout()
plt.show()

# 图像合成
print("\n=== 图像合成 ===")

# 创建两个图像
img1 = Image.new('RGB', (200, 200), color=(255, 100, 100))
img2 = Image.new('RGB', (200, 200), color=(100, 100, 255))

# 混合
print("1. 图像混合 (blend):")
blended = Image.blend(img1, img2, alpha=0.5)

# 合成
print("2. 图像合成 (composite):")
img1_paste = img1.copy()
mask = Image.new('L', (200, 200), color=128)  # 半透明掩码
composited = Image.composite(img1, img2, mask)

# 叠加
print("3. 叠加 (ImageChops):")
added = ImageChops.add(img1, img2)
subtracted = ImageChops.subtract(img1, img2)

# 差异
print("4. 差异 (difference):")
difference = ImageChops.difference(img1, img2)

# 可视化合成效果
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes[0, 0].imshow(img1)
axes[0, 0].set_title('图像1')
axes[0, 0].axis('off')

axes[0, 1].imshow(img2)
axes[0, 1].set_title('图像2')
axes[0, 1].axis('off')

axes[0, 2].imshow(blended)
axes[0, 2].set_title('混合 (alpha=0.5)')
axes[0, 2].axis('off')

axes[1, 0].imshow(composited)
axes[1, 0].set_title('合成')
axes[1, 0].axis('off')

axes[1, 1].imshow(added)
axes[1, 1].set_title('叠加')
axes[1, 1].axis('off')

axes[1, 2].imshow(difference)
axes[1, 2].set_title('差异')
axes[1, 2].axis('off')

plt.suptitle('图像合成效果', fontsize=14)
plt.tight_layout()
plt.show()

# 图像运算
print("\n=== 图像运算 ===")

# 算术运算
img_arr = np.array(img)

# 加法
added_arr = np.clip(img_arr.astype(int) + 50, 0, 255).astype(np.uint8)

# 减法
subtracted_arr = np.clip(img_arr.astype(int) - 50, 0, 255).astype(np.uint8)

# 乘法
multiplied_arr = np.clip(img_arr.astype(int) * 1.2, 0, 255).astype(np.uint8)

# 除法
divided_arr = np.clip(img_arr.astype(int) / 1.2, 0, 255).astype(np.uint8)

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(added_arr)
axes[0, 0].set_title('加法 (+50)')
axes[0, 0].axis('off')

axes[0, 1].imshow(subtracted_arr)
axes[0, 1].set_title('减法 (-50)')
axes[0, 1].axis('off')

axes[1, 0].imshow(multiplied_arr)
axes[1, 0].set_title('乘法 (x1.2)')
axes[1, 0].axis('off')

axes[1, 1].imshow(divided_arr)
axes[1, 1].set_title('除法 (/1.2)')
axes[1, 1].axis('off')

plt.suptitle('图像算术运算', fontsize=14)
plt.tight_layout()
plt.show()

# 图像通道操作
print("\n=== 图像通道操作 ===")

# 分离通道
r, g, b = img.split()

# 分别增强每个通道
r_enhanced = ImageEnhance.Brightness(r).enhance(1.5)
g_enhanced = ImageEnhance.Brightness(g).enhance(1.2)
b_enhanced = ImageEnhance.Brightness(b).enhance(0.8)

# 合并通道
enhanced_img = Image.merge('RGB', [r_enhanced, g_enhanced, b_enhanced])

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(r, cmap='Reds')
axes[0, 0].set_title('红色通道')
axes[0, 0].axis('off')

axes[0, 1].imshow(g, cmap='Greens')
axes[0, 1].set_title('绿色通道')
axes[0, 1].axis('off')

axes[1, 0].imshow(b, cmap='Blues')
axes[1, 0].set_title('蓝色通道')
axes[1, 0].axis('off')

axes[1, 1].imshow(enhanced_img)
axes[1, 1].set_title('通道增强后')
axes[1, 1].axis('off')

plt.suptitle('图像通道操作', fontsize=14)
plt.tight_layout()
plt.show()

# 图像模式转换
print("\n=== 图像模式转换 ===")

# RGB转灰度
gray = img.convert('L')
print(f"RGB转灰度: {img.mode} -> {gray.mode}")

# RGB转RGBA
rgba = img.convert('RGBA')
print(f"RGB转RGBA: {img.mode} -> {rgba.mode}")

# 灰度转二值
threshold = 128
binary = gray.point(lambda x: 255 if x > threshold else 0)
binary = binary.convert('1')
print(f"灰度转二值 (阈值={threshold})")

# 可视化模式转换
fig, axes = plt.subplots(1, 4, figsize=(15, 4))
axes[0].imshow(img)
axes[0].set_title(f'RGB模式')
axes[0].axis('off')

axes[1].imshow(gray, cmap='gray')
axes[1].set_title('灰度模式')
axes[1].axis('off')

axes[2].imshow(rgba)
axes[2].set_title('RGBA模式')
axes[2].axis('off')

axes[3].imshow(binary, cmap='gray')
axes[3].set_title('二值模式')
axes[3].axis('off')

plt.suptitle('图像模式转换', fontsize=14)
plt.tight_layout()
plt.show()

# 图像直方图
print("\n=== 图像直方图 ===")

# 计算直方图
gray_arr = np.array(gray)
histogram, bins = np.histogram(gray_arr.flatten(), bins=256, range=[0, 256])

# 计算累计直方图
cumulative_hist = np.cumsum(histogram)

# 直方图均衡化
eq_hist, eq_bins = np.histogram(gray_arr.flatten(), bins=256, range=[0, 256])
eq_cum = np.cumsum(eq_hist)
eq_cum = (eq_cum - eq_cum.min()) * 255 / (eq_cum.max() - eq_cum.min())
eq_cum = eq_cum.astype('uint8')

# 应用均衡化
equalized = gray.point(lambda x: eq_cum[x])

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].imshow(gray, cmap='gray')
axes[0, 0].set_title('原始灰度图像')
axes[0, 0].axis('off')

axes[0, 1].imshow(equalized, cmap='gray')
axes[0, 1].set_title('直方图均衡化')
axes[0, 1].axis('off')

axes[1, 0].plot(histogram, color='blue')
axes[1, 0].set_title('原始灰度直方图')
axes[1, 0].set_xlabel('灰度值')
axes[1, 0].set_ylabel('频数')

axes[1, 1].plot(eq_cum, color='green')
axes[1, 1].set_title('均衡化后直方图')
axes[1, 1].set_xlabel('灰度值')
axes[1, 1].set_ylabel('累计频数')

plt.suptitle('直方图处理', fontsize=14)
plt.tight_layout()
plt.show()

# 图像几何变换
print("\n=== 图像几何变换 ===")

# 仿射变换
print("1. 仿射变换:")
matrix = [1, 0.2, 0, 0, 1, 0]  # 倾斜
affine_img = img.transform((width, height), Image.AFFINE, matrix, Image.BILINEAR)

# 透视变换
print("2. 透视变换:")
quad = (0, 0, width, 0, width, height, 0, height)
rect = (0, 0, width, 0, width-50, height, 50, height)
perspective_img = img.transform((width, height), Image.QUAD, rect, Image.BILINEAR)

# 拉伸变换
print("3. 拉伸变换:")
stretched = img.resize((int(width * 1.5), int(height * 0.8)), Image.BILINEAR)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(affine_img)
axes[0].set_title('仿射变换')
axes[0].axis('off')

axes[1].imshow(perspective_img)
axes[1].set_title('透视变换')
axes[1].axis('off')

axes[2].imshow(stretched)
axes[2].set_title('拉伸变换')
axes[2].axis('off')

plt.suptitle('图像几何变换', fontsize=14)
plt.tight_layout()
plt.show()

# 图像蒙版和抠图
print("\n=== 图像蒙版和抠图 ===")

# 创建渐变蒙版
mask = Image.new('L', (width, height), color=0)
draw_mask = ImageDraw.Draw(mask)

# 绘制渐变
for y in range(height):
    value = int(255 * y / height)
    draw_mask.line([(0, y), (width, y)], fill=value)

# 应用蒙版
masked_img = Image.new('RGBA', (width, height), color=(255, 255, 255, 0))
masked_img.paste(img, (0, 0), mask)

# 创建椭圆蒙版
ellipse_mask = Image.new('L', (width, height), color=0)
draw_ellipse = ImageDraw.Draw(ellipse_mask)
draw_ellipse.ellipse([50, 25, width-50, height-25], fill=255)

# 应用椭圆蒙版
ellipse_masked = Image.new('RGB', (width, height), color=(200, 200, 200))
ellipse_masked.paste(img, (0, 0), ellipse_mask)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(mask, cmap='gray')
axes[0].set_title('渐变蒙版')
axes[0].axis('off')

axes[1].imshow(masked_img)
axes[1].set_title('蒙版应用效果')
axes[1].axis('off')

axes[2].imshow(ellipse_masked)
axes[2].set_title('椭圆抠图')
axes[2].axis('off')

plt.suptitle('蒙版和抠图', fontsize=14)
plt.tight_layout()
plt.show()

# 批量图像处理
print("\n=== 批量图像处理 ===")

# 创建多个测试图像
test_images = []
for i in range(4):
    test_img = Image.new('RGB', (100, 100), color=(i*50+50, (3-i)*50+50, 100))
    test_images.append(test_img)

# 创建缩略图
for i, test_img in enumerate(test_images):
    thumb = test_img.copy()
    thumb.thumbnail((50, 50))
    test_images[i] = thumb

# 创建拼贴图像
collage = Image.new('RGB', (220, 220), color=(255, 255, 255))
positions = [(0, 0), (110, 0), (0, 110), (110, 110)]

for img_item, pos in zip(test_images, positions):
    collage.paste(img_item, pos)

plt.figure(figsize=(8, 8))
plt.imshow(collage)
plt.title('图像拼贴')
plt.axis('off')
plt.show()

# 清理临时文件
import os
if os.path.exists('sample_image.jpg'):
    os.remove('sample_image.jpg')
    print("临时文件已清理")

# 总结
print("\n=== Pillow图像处理学习总结 ===")
print("1. 图像滤镜效果")
print("2. 自定义滤镜")
print("3. 图像增强（亮度、对比度、饱和度、锐度）")
print("4. 图像变换（缩放、旋转、翻转、裁剪）")
print("5. 图像合成（混合、叠加、差异）")
print("6. 图像运算（加、减、乘、除）")
print("7. 图像通道操作")
print("8. 图像模式转换")
print("9. 直方图处理")
print("10. 几何变换（仿射、透视）")
print("11. 蒙版和抠图")
print("12. 批量处理和拼贴")

print("\nPillow图像处理高级学习完成！")
