# TensorFlow TensorBoard使用学习
# 主要内容：TensorBoard可视化、日志记录

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("=== 创建模型 ===")
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("\n=== 创建TensorBoard回调 ===")
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

print("\n=== 创建模拟数据 ===")
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)

print("\n=== 训练模型 ===")
model.fit(
    X, y,
    epochs=10,
    validation_split=0.2,
    callbacks=[tensorboard_callback]
)

print("\n=== 手动记录日志 ===")
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = f'./logs/{current_time}'
summary_writer = tf.summary.create_file_writer(log_dir)

with summary_writer.as_default():
    tf.summary.scalar('custom_loss', 0.5, step=0)
    tf.summary.scalar('custom_accuracy', 0.85, step=0)

print("\n=== 清理测试文件 ===")
import shutil
import os
if os.path.exists('./logs'):
    shutil.rmtree('./logs')
    print("已删除日志目录")