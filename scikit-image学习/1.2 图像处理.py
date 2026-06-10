# scikit-image图像处理学习
# 主要内容：边缘检测、滤波、分割

from skimage import data, filters, segmentation, feature
import matplotlib.pyplot as plt

print("=== 使用内置数据集 ===")
img = data.camera()
plt.imshow(img, cmap='gray')
plt.title('原始图像')
plt.show()

print("\n=== 边缘检测 ===")
edges = filters.sobel(img)
plt.imshow(edges, cmap='gray')
plt.title('Sobel边缘检测')
plt.show()

print("\n=== 滤波操作 ===")
gaussian = filters.gaussian(img, sigma=2)
plt.imshow(gaussian, cmap='gray')
plt.title('高斯滤波')
plt.show()

print("\n=== 图像分割 ===")
segments = segmentation.slic(img, n_segments=100)
plt.imshow(segmentation.mark_boundaries(img, segments))
plt.title('SLIC分割')
plt.show()

print("\n=== 特征检测 ===")
corners = feature.corner_harris(img)
plt.imshow(corners, cmap='gray')
plt.title('Harris角点检测')
plt.show()

print("\n=== 阈值处理 ===")
threshold = filters.threshold_otsu(img)
binary = img > threshold
plt.imshow(binary, cmap='gray')
plt.title('Otsu阈值分割')
plt.show()