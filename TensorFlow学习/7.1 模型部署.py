# TensorFlow模型部署学习
# 主要内容：SavedModel、TensorFlow Serving、TensorFlow Lite、模型量化

import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

print("=== 创建模型 ===")
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("模型已创建")

print("\n=== 训练模型 ===")
X_train = np.random.randn(1000, 20)
y_train = np.random.randint(0, 10, 1000)

model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=1)

print("\n=== SavedModel格式 ===")
saved_model_path = './saved_model'
model.save(saved_model_path)
print(f"模型已保存到: {saved_model_path}")

print("\n=== 加载SavedModel ===")
loaded_model = tf.saved_model.load(saved_model_path)
print("SavedModel已加载")

print("\n=== 检查模型签名 ===")
infer = loaded_model.signatures['serving_default']
print(f"输入签名: {infer.structured_input_signature}")
print(f"输出签名: {infer.structured_outputs}")

print("\n=== HDF5格式 ===")
h5_path = './model.h5'
model.save(h5_path)
print(f"HDF5模型已保存到: {h5_path}")

loaded_h5 = keras.models.load_model(h5_path)
print("HDF5模型已加载")

print("\n=== TensorFlow Lite转换 ===")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

tflite_path = './model.tflite'
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

print(f"TFLite模型已保存到: {tflite_path}")
print(f"TFLite模型大小: {os.path.getsize(tflite_path) / 1024:.2f} KB")

print("\n=== TFLite推理 ===")
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.random.randn(1, 20).astype(np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print(f"TFLite输出形状: {output_data.shape}")
print(f"预测类别: {np.argmax(output_data)}")

print("\n=== 模型量化 ===")
converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]

def representative_dataset():
    for _ in range(100):
        yield [np.random.randn(1, 20).astype(np.float32)]

converter_quant.representative_dataset = representative_dataset
converter_quant.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_quant.inference_input_type = tf.int8
converter_quant.inference_output_type = tf.int8

tflite_quant = converter_quant.convert()

quant_path = './model_quant.tflite'
with open(quant_path, 'wb') as f:
    f.write(tflite_quant)

print(f"量化模型大小: {os.path.getsize(quant_path) / 1024:.2f} KB")

print("\n=== TensorFlow.js格式 ===")
print("使用tensorflowjs_converter转换:")
print("  tensorflowjs_converter --input_format=tf_saved_model ./saved_model ./tfjs_model")

print("\n=== 模型大小对比 ===")
print(f"SavedModel大小: {sum(os.path.getsize(os.path.join(dirpath, filename)) for dirpath, dirnames, filenames in os.walk(saved_model_path) for filename in filenames) / 1024:.2f} KB")
print(f"HDF5大小: {os.path.getsize(h5_path) / 1024:.2f} KB")
print(f"TFLite大小: {os.path.getsize(tflite_path) / 1024:.2f} KB")
print(f"量化TFLite大小: {os.path.getsize(quant_path) / 1024:.2f} KB")

print("\n=== 清理 ===")
import shutil
for path in [saved_model_path, h5_path, tflite_path, quant_path]:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"已删除: {path}")