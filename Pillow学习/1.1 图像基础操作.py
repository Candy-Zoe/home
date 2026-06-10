# Pillow图像基础操作学习
# 主要内容：图像读取、显示、保存、基本属性

from PIL import Image
import matplotlib.pyplot as plt

print("=== 读取图像 ===")
img = Image.open('example.jpg')
print(f"图像格式: {img.format}")
print(f"图像大小: {img.size}")
print(f"图像模式: {img.mode}")

print("\n=== 显示图像 ===")
plt.imshow(img)
plt.axis('off')
plt.show()

print("\n=== 保存图像 ===")
img.save('saved_image.png')
print("图像已保存为PNG格式")

print("\n=== 创建新图像 ===")
new_img = Image.new('RGB', (200, 100), color='red')
plt.imshow(new_img)
plt.title('红色图像')
plt.show()

print("\n=== 图像模式转换 ===")
gray_img = img.convert('L')
plt.imshow(gray_img, cmap='gray')
plt.title('灰度图像')
plt.show()

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('saved_image.png'):
    os.remove('saved_image.png')
    print("已删除测试文件")