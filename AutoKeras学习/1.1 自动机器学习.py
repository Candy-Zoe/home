# AutoKeras自动机器学习学习
# 主要内容：分类任务、回归任务、自动搜索

import autokeras as ak
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

print("=== 分类任务 ===")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = ak.StructuredDataClassifier(max_trials=3)
clf.fit(X_train, y_train, epochs=10)
accuracy = clf.evaluate(X_test, y_test)[1]
print(f"分类准确率: {accuracy:.4f}")

print("\n=== 回归任务 ===")
from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg = ak.StructuredDataRegressor(max_trials=3)
reg.fit(X_train, y_train, epochs=10)
mse = reg.evaluate(X_test, y_test)[1]
print(f"回归MSE: {mse:.4f}")

print("\n=== 图像分类 ===")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

image_clf = ak.ImageClassifier(max_trials=2)
image_clf.fit(x_train, y_train, epochs=3)
accuracy = image_clf.evaluate(x_test, y_test)[1]
print(f"图像分类准确率: {accuracy:.4f}")

print("\n=== 获取最佳模型 ===")
model = image_clf.export_model()
print(f"模型摘要:\n{model.summary()}")

print("\n=== 清理数据 ===")
import shutil
import os
if os.path.exists('./data'):
    shutil.rmtree('./data')
if os.path.exists('./autokeras'):
    shutil.rmtree('./autokeras')
print("已删除数据目录")