# Pillow图像合成与水印学习
# 主要内容：图像合成、添加水印、文字叠加

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

print("=== 创建底图 ===")
base_img = Image.new('RGB', (400, 300), color='lightblue')
plt.imshow(base_img)
plt.title('底图')
plt.show()

print("\n=== 添加文字 ===")
draw = ImageDraw.Draw(base_img)
font = ImageFont.truetype('arial.ttf', 36)
draw.text((100, 120), 'Hello Pillow!', fill='red', font=font)
plt.imshow(base_img)
plt.title('添加文字')
plt.show()

print("\n=== 添加形状 ===")
draw.rectangle([50, 50, 150, 100], fill='green', outline='blue')
draw.ellipse([250, 50, 350, 100], fill='yellow', outline='red')
plt.imshow(base_img)
plt.title('添加形状')
plt.show()

print("\n=== 添加水印 ===")
watermark = Image.new('RGBA', base_img.size, (255, 255, 255, 0))
watermark_draw = ImageDraw.Draw(watermark)
watermark_draw.text((150, 250), 'Watermark', fill=(255, 0, 0, 128), font=font)

watermarked = Image.composite(watermark, base_img, watermark)
plt.imshow(watermarked)
plt.title('添加水印')
plt.show()

print("\n=== 图像合成 ===")
small_img = Image.new('RGB', (100, 100), color='purple')
base_img.paste(small_img, (200, 180))
plt.imshow(base_img)
plt.title('图像合成')
plt.show()

print("\n=== 透明度处理 ===")
alpha_img = Image.new('RGBA', (200, 200), (255, 0, 0, 128))
base_rgba = base_img.convert('RGBA')
combined = Image.alpha_composite(base_rgba, alpha_img)
plt.imshow(combined)
plt.title('透明度处理')
plt.show()