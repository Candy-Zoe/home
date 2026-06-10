# TensorFlow Keras高级API学习
# 主要内容：自定义回调、回调链、EarlyStopping、ModelCheckpoint高级用法

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建模型 ===")
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10)
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n=== 自定义回调 ===")
class CustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Epoch {epoch} 开始")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch} 结束 - loss: {logs['loss']:.4f}, acc: {logs['accuracy']:.4f}")
    
    def on_batch_begin(self, batch, logs=None):
        pass
    
    def on_batch_end(self, batch, logs=None):
        if batch % 100 == 0:
            print(f"Batch {batch} 结束 - loss: {logs['loss']:.4f}")

print("自定义回调已定义")

print("\n=== LearningRateScheduler ===")
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95

lr_callback = keras.callbacks.LearningRateScheduler(scheduler)
print("学习率调度器已创建")

print("\n=== ReduceLROnPlateau ===")
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
print("学习率降低回调已创建")

print("\n=== TensorBoard ===")
tensorboard = keras.callbacks.TensorBoard(
    log_dir='./logs',
    histogram_freq=1,
    write_graph=True
)
print("TensorBoard回调已创建")

print("\n=== CSVLogger ===")
csv_logger = keras.callbacks.CSVLogger('training.log')
print("CSV日志回调已创建")

print("\n=== LambdaCallback ===")
lambda_callback = keras.callbacks.LambdaCallback(
    on_epoch_end=lambda epoch, logs: print(f"Epoch {epoch} val_loss: {logs.get('val_loss', 0):.4f}")
)
print("Lambda回调已创建")

print("\n=== 组合回调 ===")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.h5',
        monitor='val_loss',
        save_best_only=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    ),
    CustomCallback()
]

print(f"回调数量: {len(callbacks)}")

print("\n=== 训练模型 ===")
X = np.random.randn(1000, 20)
y = np.random.randint(0, 10, 1000)

history = model.fit(
    X, y,
    epochs=20,
    validation_split=0.2,
    callbacks=callbacks[:1],
    verbose=0
)

print("训练完成")

print("\n=== 绘制训练历史 ===")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='训练损失')
plt.plot(history.history['val_loss'], label='验证损失')
plt.title('损失曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='训练准确率')
plt.plot(history.history['val_accuracy'], label='验证准确率')
plt.title('准确率曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

print("\n=== 清理 ===")
import os
for f in ['best_model.h5', 'training.log']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除 {f}")