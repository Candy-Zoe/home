# TensorFlow线性回归学习
# 主要内容：使用TensorFlow实现线性回归

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建模拟数据 ===")
np.random.seed(42)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * x + 1 + np.random.randn(100, 1) * 0.5

plt.scatter(x, y)
plt.title('模拟线性数据')
plt.show()

print("\n=== 定义模型 ===")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.summary()

print("\n=== 编译模型 ===")
model.compile(optimizer='sgd', loss='mse')

print("\n=== 训练模型 ===")
history = model.fit(x, y, epochs=100, verbose=0)

print("\n=== 训练结果 ===")
weights, bias = model.layers[0].get_weights()
print(f"学习到的权重: {weights[0][0]:.4f}")
print(f"学习到的偏置: {bias[0]:.4f}")

plt.plot(history.history['loss'])
plt.title('训练损失')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

print("\n=== 预测 ===")
y_pred = model.predict(x)

plt.scatter(x, y, label='数据点')
plt.plot(x, y_pred, 'r-', label='拟合直线')
plt.legend()
plt.title('线性回归拟合结果')
plt.show()