# TensorFlow模型保存与加载学习
# 主要内容：模型保存、模型加载、检查点

import tensorflow as tf
import numpy as np

print("=== 定义简单模型 ===")
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')

print(f"原始模型参数:\n{model.layers[0].get_weights()[0][0]}")

print("\n=== 保存整个模型 ===")
model.save('my_model')
print("模型已保存")

print("\n=== 加载模型 ===")
loaded_model = tf.keras.models.load_model('my_model')
print(f"加载后模型参数:\n{loaded_model.layers[0].get_weights()[0][0]}")

print("\n=== 保存为HDF5格式 ===")
model.save('my_model.h5')
loaded_h5 = tf.keras.models.load_model('my_model.h5')
print("HDF5模型已加载")

print("\n=== 保存检查点 ===")
checkpoint_path = "training_checkpoints/cp.ckpt"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

x_train = np.random.rand(100, 5)
y_train = tf.keras.utils.to_categorical(np.random.randint(0, 5, 100), 5)

model.fit(x_train, y_train, epochs=3, callbacks=[cp_callback])

print("\n=== 从检查点恢复 ===")
new_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(5, activation='softmax')
])

new_model.load_weights(checkpoint_path)
print("权重已加载")

print("\n=== 清理测试文件 ===")
import shutil
import os
if os.path.exists('my_model'):
    shutil.rmtree('my_model')
    print("已删除: my_model")
if os.path.exists('my_model.h5'):
    os.remove('my_model.h5')
    print("已删除: my_model.h5")
if os.path.exists('training_checkpoints'):
    shutil.rmtree('training_checkpoints')
    print("已删除: training_checkpoints")