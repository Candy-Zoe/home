# Pillow图像变换学习
# 主要内容：裁剪、缩放、旋转、翻转

from PIL import Image
import matplotlib.pyplot as plt

print("=== 裁剪图像 ===")
img = Image.open('example.jpg')
cropped = img.crop((100, 100, 300, 300))
plt.imshow(cropped)
plt.title('裁剪后的图像')
plt.show()

print("\n=== 缩放图像 ===")
resized = img.resize((200, 200))
print(f"原始大小: {img.size}")
print(f"缩放后大小: {resized.size}")
plt.imshow(resized)
plt.title('缩放后的图像')
plt.show()

print("\n=== 旋转图像 ===")
rotated = img.rotate(45)
plt.imshow(rotated)
plt.title('旋转45度')
plt.show()

rotated_expand = img.rotate(45, expand=True)
plt.imshow(rotated_expand)
plt.title('旋转45度并扩展')
plt.show()

print("\n=== 翻转图像 ===")
flipped_h = img.transpose(Image.FLIP_LEFT_RIGHT)
plt.imshow(flipped_h)
plt.title('水平翻转')
plt.show()

flipped_v = img.transpose(Image.FLIP_TOP_BOTTOM)
plt.imshow(flipped_v)
plt.title('垂直翻转')
plt.show()