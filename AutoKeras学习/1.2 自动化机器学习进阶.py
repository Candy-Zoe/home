# AutoKeras自动化机器学习进阶学习
# 主要内容：自定义搜索空间、图像分类进阶、文本分类、自定义回调

import autokeras as ak
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建示例数据 ===")
from sklearn.datasets import load_breast_cancer, load_iris

data = load_breast_cancer()
X = data.data
y = data.target

print(f"特征数: {X.shape[1]}")
print(f"样本数: {X.shape[0]}")

print("\n=== StructuredDataRegressor ===")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

reg = ak.StructuredDataRegressor(
    max_trials=3,
    directory="autokeras_reg",
    overwrite=True
)

reg.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
print("回归模型训练完成")

print("\n=== 模型评估 ===")
predicted_y = reg.predict(X_test)
print(f"预测结果前5个: {predicted_y[:5].flatten()}")

print("\n=== 获取最佳模型 ===")
model = reg.export_model()
print(f"最佳模型结构:\n{model.summary()}")

print("\n=== 自定义搜索空间 ===")
from autokeras import AutoModel
from tensorflow import keras

inputs = ak.StructuredDataInput()
outputs = ak.DenseBlock()(inputs)
outputs = ak.RegressionHead()(outputs)
custom_model = AutoModel(inputs=inputs, outputs=outputs, max_trials=3, directory="autokeras_custom")
print("自定义模型已创建")

print("\n=== 图像分类 ===")
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

clf = ak.ImageClassifier(
    max_trials=2,
    directory="autokeras_img",
    overwrite=True
)

clf.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
print("图像分类模型训练完成")

print("\n=== 图像分类预测 ===")
predicted = clf.predict(x_test[:5])
print(f"预测类别: {predicted.flatten()}")
print(f"真实类别: {y_test[:5]}")

print("\n=== 文本分类 ===")
from tensorflow.keras.datasets import imdb

vocab_size = 10000
max_length = 200

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=vocab_size)

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, maxlen=max_length)
x_test = pad_sequences(x_test, maxlen=max_length)

text_clf = ak.TextClassifier(
    max_trials=2,
    directory="autokeras_text",
    overwrite=True
)

text_clf.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test))
print("文本分类模型训练完成")

print("\n=== 自定义回调 ===")
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

clf_with_callback = ak.ImageClassifier(
    max_trials=2,
    directory="autokeras_callback",
    overwrite=True,
    callbacks=[early_stopping]
)

print("自定义回调已添加")

print("\n=== 模型调优 ===")
from keras_tuner import BayesianOptimization

tuner = BayesianOptimization(
    ak.ImageClassifier(max_trials=2),
    objective='val_accuracy',
    max_trials=3,
    directory="autokeras_tuner"
)

tuner.search(x_train[:1000], y_train[:1000], epochs=2, validation_data=(x_test[:500], y_test[:500]))
print("超参数调优完成")

print("\n=== 清理测试文件 ===")
import shutil
import os
for path in ["autokeras_reg", "autokeras_custom", "autokeras_img", "autokeras_text", "autokeras_callback", "autokeras_tuner"]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"已删除 {path}")