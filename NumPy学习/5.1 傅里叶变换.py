# NumPy傅里叶变换学习
# 主要内容：一维FFT、二维FFT、频谱分析

import numpy as np
import matplotlib.pyplot as plt

print("=== 一维傅里叶变换 ===")
t = np.linspace(0, 1, 1000, endpoint=False)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t)

plt.plot(t, signal)
plt.title('原始信号')
plt.show()

print("\n=== FFT变换 ===")
fft_result = np.fft.fft(signal)
freq = np.fft.fftfreq(len(t), t[1] - t[0])

plt.plot(freq, np.abs(fft_result))
plt.title('频谱')
plt.xlim(0, 20)
plt.show()

print("\n=== 逆FFT ===")
ifft_result = np.fft.ifft(fft_result)
plt.plot(t, np.real(ifft_result))
plt.title('逆变换恢复信号')
plt.show()

print("\n=== 二维FFT ===")
from scipy import misc
face = misc.face(gray=True)
fft2_result = np.fft.fft2(face)
fft2_shift = np.fft.fftshift(fft2_result)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(face, cmap='gray')
plt.title('原始图像')

plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(fft2_shift)), cmap='gray')
plt.title('二维频谱')
plt.show()