# TensorFlow神经网络分类学习
# 主要内容：使用TensorFlow实现全连接神经网络进行MNIST手写数字分类

import tensorflow as tf
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

print("=== 加载MNIST数据集 ===")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"训练集形状: {x_train.shape}")
print(f"测试集形状: {x_test.shape}")

plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_train[i], cmap='gray')
    plt.title(f"标签: {y_train[i]}")
    plt.axis('off')
plt.show()

print("\n=== 数据预处理 ===")
x_train = x_train / 255.0
x_test = x_test / 255.0

print("\n=== 定义模型 ===")
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()

print("\n=== 编译模型 ===")
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== 训练模型 ===")
history = model.fit(x_train, y_train, epochs=5, validation_split=0.1)

print("\n=== 评估模型 ===")
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"测试准确率: {test_acc:.4f}")

print("\n=== 可视化训练结果 ===")
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.legend()
plt.show()

print("\n=== 预测 ===")
predictions = model.predict(x_test)
plt.figure(figsize=(8, 8))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"预测: {predictions[i].argmax()}, 真实: {y_test[i]}")
    plt.axis('off')
plt.show()