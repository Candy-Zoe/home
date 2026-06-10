# TensorFlow自定义训练循环学习
# 主要内容：自定义训练循环、梯度裁剪、学习率调度

import tensorflow as tf
import numpy as np

print("=== 创建数据集 ===")
X = np.random.rand(100, 2)
y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

print("\n=== 定义模型 ===")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(1)
])

print("\n=== 自定义训练循环 ===")
loss_fn = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

epochs = 100
loss_history = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        y_pred = model(X, training=True)
        loss = loss_fn(y, y_pred)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    loss_history.append(loss.numpy())
    
    if (epoch + 1) % 20 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.numpy():.4f}")

import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.title('训练损失')
plt.show()

print("\n=== 学习率调度 ===")
initial_lr = 0.1
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_lr, decay_steps=1000, decay_rate=0.96, staircase=True
)

optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule)

for step in range(5):
    current_lr = lr_schedule(step).numpy()
    print(f"Step {step}, 学习率: {current_lr:.6f}")