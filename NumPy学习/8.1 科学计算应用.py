# NumPy科学计算应用学习
# 主要内容：插值、积分、微分方程求解、信号处理

import numpy as np
from scipy import interpolate, integrate, signal, optimize
import matplotlib.pyplot as plt

print("=== 插值 ===")
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([0, 0.8, 0.9, 0.1, -0.8, -1])

x_new = np.linspace(0, 5, 100)

linear_interp = interpolate.interp1d(x, y, kind='linear')
cubic_interp = interpolate.interp1d(x, y, kind='cubic')

plt.figure(figsize=(10, 5))
plt.scatter(x, y, label='数据点', color='red', s=100)
plt.plot(x_new, linear_interp(x_new), label='线性插值')
plt.plot(x_new, cubic_interp(x_new), label='三次样条插值')
plt.legend()
plt.title('插值方法比较')
plt.show()

print("\n=== 数值积分 ===")
def f(x):
    return np.sin(x)

result, error = integrate.quad(f, 0, np.pi)
print(f"∫sin(x)dx from 0 to π = {result:.6f}")
print(f"误差估计: {error:.2e}")

print("\n=== 多重积分 ===")
def f2(x, y):
    return x * y

result = integrate.dblquad(f2, 0, 1, lambda x: 0, lambda x: 1)
print(f"∫∫xy dA = {result[0]:.6f}")

print("\n=== 常微分方程求解 ===")
def dydt(y, t):
    return -2 * y

t = np.linspace(0, 5, 100)
y0 = 1.0

from scipy.integrate import odeint
solution = odeint(dydt, y0, t)

plt.figure(figsize=(10, 5))
plt.plot(t, solution, label='数值解')
plt.plot(t, y0 * np.exp(-2 * t), '--', label='解析解')
plt.legend()
plt.title('ODE求解: dy/dt = -2y')
plt.show()

print("\n=== 信号处理 ===")
fs = 1000
t = np.linspace(0, 1, fs)
signal_data = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

fft_result = np.fft.fft(signal_data)
freqs = np.fft.fftfreq(len(t), 1/fs)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t[:100], signal_data[:100])
plt.title('时域信号')
plt.xlabel('时间 (s)')

plt.subplot(1, 2, 2)
plt.plot(freqs[:len(freqs)//2], np.abs(fft_result[:len(fft_result)//2]))
plt.title('频域信号')
plt.xlabel('频率 (Hz)')
plt.tight_layout()
plt.show()

print("\n=== 滤波器设计 ===")
b, a = signal.butter(4, 0.1, 'low')
filtered = signal.filtfilt(b, a, signal_data)

plt.figure(figsize=(10, 5))
plt.plot(t[:200], signal_data[:200], label='原始信号')
plt.plot(t[:200], filtered[:200], label='滤波后')
plt.legend()
plt.title('低通滤波')
plt.show()

print("\n=== 优化求解 ===")
def objective(x):
    return x**2 + 10*np.sin(x)

result = optimize.minimize(objective, x0=0)
print(f"最小值位置: x = {result.x[0]:.4f}")
print(f"最小值: f(x) = {result.fun:.4f}")

print("\n=== 方程求根 ===")
def equation(x):
    return x**3 - 2*x - 5

root = optimize.brentq(equation, 2, 3)
print(f"方程根: x = {root:.6f}")

print("\n=== 曲线拟合 ===")
x_data = np.linspace(0, 10, 50)
y_data = 2.5 * np.sin(1.5 * x_data) + 0.5 * np.random.randn(50)

def fit_func(x, a, b):
    return a * np.sin(b * x)

params, _ = optimize.curve_fit(fit_func, x_data, y_data, p0=[2, 1])
print(f"拟合参数: a = {params[0]:.4f}, b = {params[1]:.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(x_data, y_data, label='数据', alpha=0.6)
plt.plot(x_data, fit_func(x_data, *params), 'r-', label='拟合曲线')
plt.legend()
plt.title('曲线拟合')
plt.show()