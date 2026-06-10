# scikit-learn模型评估学习
# 主要内容：交叉验证、混淆矩阵、分类指标、回归指标

import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score
)
import matplotlib.pyplot as plt
import seaborn as sns

print("=== 分类模型评估 ===")
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"召回率: {recall_score(y_test, y_pred, average='macro'):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('混淆矩阵')
plt.show()

print("\n=== 交叉验证 ===")
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"5折交叉验证分数: {scores}")
print(f"平均准确率: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

print("\n=== 回归模型评估 ===")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)

print(f"均方误差 (MSE): {mean_squared_error(y_test, y_pred):.4f}")
print(f"R²分数: {r2_score(y_test, y_pred):.4f}")