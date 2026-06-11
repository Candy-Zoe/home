# Pillow图像基础操作学习
# 主要内容：图像打开、保存、属性查看、基础变换

# 导入Pillow库
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# 创建测试图像
print("=== 创建测试图像 ===")

# 创建一个简单的RGB图像
width, height = 400, 300
img = Image.new('RGB', (width, height), color=(255, 0, 0))  # 红色背景

# 在图像上绘制一些内容
from PIL import ImageDraw, ImageFont

draw = ImageDraw.Draw(img)

# 绘制一个矩形
draw.rectangle([50, 50, 200, 150], fill=(0, 255, 0), outline=(255, 255, 0), width=3)

# 绘制一个椭圆
draw.ellipse([220, 50, 370, 150], fill=(0, 0, 255), outline=(255, 255, 255), width=3)

# 绘制文字
try:
    # 尝试使用系统字体
    font = ImageFont.truetype("arial.ttf", 30)
except:
    # 如果找不到，使用默认字体
    font = ImageFont.load_default()

draw.text((100, 180), "Pillow Demo", fill=(255, 255, 0), font=font)

print(f"测试图像已创建: {width}x{height}")

# 保存测试图像
img.save('test_pillow.png')

# 图像属性
print("\n=== 图像属性 ===")

# 打开图像
img = Image.open('test_pillow.png')

print(f"图像格式: {img.format}")
print(f"图像模式: {img.mode}")  # RGB, RGBA, L(灰度), CMYK等
print(f"图像尺寸: {img.size}")  # (宽度, 高度)
print(f"图像宽度: {img.width}")
print(f"图像高度: {img.height}")

# 图像的基本操作
print("\n=== 图像基本操作 ===")

# 1. 裁剪图像
box = (50, 50, 200, 200)  # (左, 上, 右, 下)
cropped = img.crop(box)
print(f"裁剪后的尺寸: {cropped.size}")

# 2. 调整大小
resized = img.resize((200, 150))
print(f"调整后的尺寸: {resized.size}")

# 3. 旋转图像
rotated = img.rotate(45)
print(f"旋转45度后的尺寸: {rotated.size}")

# 4. 翻转图像
flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)  # 水平翻转
flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)  # 垂直翻转
print("图像翻转完成")

# 5. 转换为灰度
gray = img.convert('L')
print(f"灰度图像模式: {gray.mode}")

# 颜色模式转换
print("\n=== 颜色模式转换 ===")

# RGB转RGBA
rgba = img.convert('RGBA')
print(f"RGBA模式: {rgba.mode}")

# RGB转HSV
# Pillow没有直接的RGB到HSV转换，需要使用其他库
print("HSV转换需要使用OpenCV等其他库")

# 分离和合并通道
print("\n=== 通道操作 ===")

# 分离RGB通道
r, g, b = img.split()
print(f"R通道范围: {r.getextrema()}")
print(f"G通道范围: {g.getextrema()}")
print(f"B通道范围: {b.getextrema()}")

# 合并通道
merged = Image.merge('RGB', (b, g, r))  # 交换R和B通道
print("通道合并完成")

# 图像增强
print("\n=== 图像增强 ===")

from PIL import ImageEnhance

# 创建原始图像的副本
enhanced_img = img.copy()

# 1. 亮度调整
brightness = ImageEnhance.Brightness(enhanced_img)
bright_img = brightness.enhance(1.5)  # 增强50%
print("亮度调整完成")

# 2. 对比度调整
contrast = ImageEnhance.Contrast(enhanced_img)
contrast_img = contrast.enhance(1.3)  # 增强30%
print("对比度调整完成")

# 3. 色彩饱和度调整
color = ImageEnhance.Color(enhanced_img)
color_img = color.enhance(1.5)  # 增强50%
print("饱和度调整完成")

# 4. 锐度调整
sharpness = ImageEnhance.Sharpness(enhanced_img)
sharp_img = sharpness.enhance(2.0)  # 锐度加倍
print("锐度调整完成")

# 图像滤镜
print("\n=== 图像滤镜 ===")

from PIL import ImageFilter

# 应用不同的滤镜
blur_img = img.filter(ImageFilter.BLUR)  # 模糊
contour_img = img.filter(ImageFilter.CONTOUR)  # 轮廓
detail_img = img.filter(ImageFilter.DETAIL)  # 细节
edge_enhance_img = img.filter(ImageFilter.EDGE_ENHANCE)  # 边缘增强
emboss_img = img.filter(ImageFilter.EMBOSS)  # 浮雕
sharpen_img = img.filter(ImageFilter.SHARPEN)  # 锐化
smooth_img = img.filter(ImageFilter.SMOOTH)  # 平滑

print("滤镜应用完成")

# 点操作
print("\n=== 点操作 ===")

# 对图像的每个像素应用函数
from PIL import ImageMath

# 使用point方法调整像素值
# 将图像亮度提升1.2倍
adjusted = img.point(lambda x: int(x * 1.2))
print("点操作完成")

# 图像统计
print("\n=== 图像统计 ===")

# 将图像转换为numpy数组进行分析
img_array = np.array(img)
print(f"图像数组形状: {img_array.shape}")
print(f"数据类型: {img_array.dtype}")
print(f"像素值范围: [{img_array.min()}, {img_array.max()}]")
print(f"各通道均值: R={img_array[:,:,0].mean():.1f}, G={img_array[:,:,1].mean():.1f}, B={img_array[:,:,2].mean():.1f}")

# 直方图
print("\n=== 直方图 ===")

# 获取直方图
histogram = img.histogram()
print(f"直方图长度: {len(histogram)}")
print(f"RGB直方图: {len(histogram)//3} 个值每通道")

# 可视化
print("\n=== 可视化 ===")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 原始图像
axes[0, 0].imshow(img)
axes[0, 0].set_title('原始图像')
axes[0, 0].axis('off')

# 裁剪后的图像
axes[0, 1].imshow(cropped)
axes[0, 1].set_title('裁剪')
axes[0, 1].axis('off')

# 旋转后的图像
axes[0, 2].imshow(rotated)
axes[0, 2].set_title('旋转45度')
axes[0, 2].axis('off')

# 灰度图像
axes[1, 0].imshow(gray, cmap='gray')
axes[1, 0].set_title('灰度')
axes[1, 0].axis('off')

# 锐化图像
axes[1, 1].imshow(sharpen_img)
axes[1, 1].set_title('锐化')
axes[1, 1].axis('off')

# 亮度调整
axes[1, 2].imshow(bright_img)
axes[1, 2].set_title('亮度增强')
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# 清理测试文件
import os
if os.path.exists('test_pillow.png'):
    os.remove('test_pillow.png')
    print("\n测试文件已清理")