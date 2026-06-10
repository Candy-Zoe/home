# scikit-image高级图像处理学习
# 主要内容：图像分割进阶、形态学操作、特征提取、图像恢复

from skimage import data, filters, feature, segmentation, morphology, restoration
from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np

print("=== 加载图像 ===")
image = data.camera()
imshow(image)
plt.title('原始图像')
plt.show()

print("\n=== 边缘检测进阶 ===")
sobel = filters.sobel(image)
canny = feature.canny(image, sigma=1.0)
prewitt = filters.prewitt(image)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(sobel, cmap='gray')
plt.title('Sobel')

plt.subplot(1, 3, 2)
plt.imshow(canny, cmap='gray')
plt.title('Canny')

plt.subplot(1, 3, 3)
plt.imshow(prewitt, cmap='gray')
plt.title('Prewitt')
plt.show()

print("\n=== 阈值分割进阶 ===")
otsu = filters.threshold_otsu(image)
li = filters.threshold_li(image)
yen = filters.threshold_yen(image)

print(f"Otsu阈值: {otsu}")
print(f"Li阈值: {li}")
print(f"Yen阈值: {yen}")

print("\n=== 形态学操作 ===")
binary = image > otsu
eroded = morphology.binary_erosion(binary, morphology.disk(2))
dilated = morphology.binary_dilation(binary, morphology.disk(2))
opened = morphology.binary_opening(binary, morphology.disk(2))
closed = morphology.binary_closing(binary, morphology.disk(2))

plt.figure(figsize=(15, 5))
plt.subplot(1, 4, 1)
plt.imshow(eroded, cmap='gray')
plt.title('腐蚀')

plt.subplot(1, 4, 2)
plt.imshow(dilated, cmap='gray')
plt.title('膨胀')

plt.subplot(1, 4, 3)
plt.imshow(opened, cmap='gray')
plt.title('开运算')

plt.subplot(1, 4, 4)
plt.imshow(closed, cmap='gray')
plt.title('闭运算')
plt.show()

print("\n=== 分水岭分割 ===")
from skimage.segmentation import watershed
from scipy import ndimage

distance = ndimage.distance_transform_edt(binary)
local_max = feature.peak_local_max(distance, indices=False, footprint=np.ones((3, 3)))
markers = ndimage.label(local_max)[0]

labels = watershed(-distance, markers, mask=binary)
print(f"分割区域数: {len(np.unique(labels))}")

print("\n=== SLIC超像素分割 ===")
from skimage.segmentation import slic

coins = data.coins()
segments = slic(coins, n_segments=100, compactness=10)
print(f"超像素数: {len(np.unique(segments))}")

print("\n=== 局部二值模式(LBP) ===")
lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
print(f"LBP直方图bins: {int(lbp.max() + 1)}")

print("\n=== HOG特征 ===")
hog_features, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                       cells_per_block=(2, 2), visualize=True)
print(f"HOG特征维度: {hog_features.shape}")

print("\n=== 图像去噪 ===")
noisy = image + 0.1 * np.random.randn(*image.shape)
denoised_bilateral = restoration.denoise_bilateral(image, sigma_color=0.05, sigma_spatial=15)
denoised_tv = restoration.denoise_tv_chambolle(image, weight=0.1)

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('原始')

plt.subplot(1, 3, 2)
plt.imshow(denoised_bilateral, cmap='gray')
plt.title('双边滤波')

plt.subplot(1, 3, 3)
plt.imshow(denoised_tv, cmap='gray')
plt.title('全变分去噪')
plt.show()