# scikit-learn异常检测学习
# 主要内容：Isolation Forest、One-Class SVM、Local Outlier Factor

from sklearn.datasets import make_blobs
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

print("=== 创建含异常值的数据 ===")
X, _ = make_blobs(n_samples=200, centers=1, cluster_std=0.5, random_state=42)
X_outliers = np.random.uniform(low=-6, high=6, size=(20, 2))
X = np.r_[X, X_outliers]
y = np.array([0] * 200 + [1] * 20)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.title('含异常值的数据')
plt.show()

print("\n=== Isolation Forest ===")
clf = IsolationForest(contamination=0.1, random_state=42)
y_pred = clf.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Isolation Forest检测结果')
plt.show()

print("\n=== One-Class SVM ===")
clf = OneClassSVM(nu=0.1, kernel='rbf', gamma='auto')
y_pred = clf.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('One-Class SVM检测结果')
plt.show()

print("\n=== Local Outlier Factor ===")
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis')
plt.title('Local Outlier Factor检测结果')
plt.show()

import numpy as np