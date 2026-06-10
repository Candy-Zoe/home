# OpenCV机器学习模块学习
# 主要内容：KNN分类、SVM分类、决策树、随机森林

import cv2
import matplotlib.pyplot as plt
import numpy as np

print("=== 创建数据集 ===")
np.random.seed(42)

class1 = np.random.randn(50, 2) + np.array([-2, -2])
class2 = np.random.randn(50, 2) + np.array([2, 2])
class3 = np.random.randn(50, 2) + np.array([0, 2])

data = np.vstack((class1, class2, class3)).astype(np.float32)
labels = np.array([0]*50 + [1]*50 + [2]*50)

plt.scatter(class1[:, 0], class1[:, 1], c='r', label='Class 0')
plt.scatter(class2[:, 0], class2[:, 1], c='g', label='Class 1')
plt.scatter(class3[:, 0], class3[:, 1], c='b', label='Class 2')
plt.legend()
plt.title('数据集')
plt.show()

print("\n=== KNN分类 ===")
knn = cv2.ml.KNearest_create()
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

test_point = np.array([[0, 0]], dtype=np.float32)
ret, results, neighbors, dist = knn.findNearest(test_point, k=3)
print(f"KNN预测结果: {results[0][0]}")
print(f"最近邻: {neighbors[0]}")
print(f"距离: {dist[0]}")

print("\n=== SVM分类 ===")
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.setC(10)
svm.setGamma(1)

svm.train(data, cv2.ml.ROW_SAMPLE, labels)
ret, result = svm.predict(test_point)
print(f"SVM预测结果: {result[0][0]}")

print("\n=== 决策树 ===")
dtree = cv2.ml.DTrees_create()
dtree.setMaxDepth(10)
dtree.train(data, cv2.ml.ROW_SAMPLE, labels)
ret, result = dtree.predict(test_point)
print(f"决策树预测结果: {result[0][0]}")

print("\n=== 训练集准确率 ===")
ret, results = knn.predict(data)
knn_acc = (results.ravel() == labels).mean()
print(f"KNN准确率: {knn_acc:.4f}")

ret, results = svm.predict(data)
svm_acc = (results.ravel() == labels).mean()
print(f"SVM准确率: {svm_acc:.4f}")

ret, results = dtree.predict(data)
dtree_acc = (results.ravel() == labels).mean()
print(f"决策树准确率: {dtree_acc:.4f}")

print("\n=== 绘制决策边界 ===")
h = 0.02
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

grid_points = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
ret, predictions = svm.predict(grid_points)
predictions = predictions.reshape(xx.shape)

plt.contourf(xx, yy, predictions, alpha=0.4)
plt.scatter(data[:, 0], data[:, 1], c=labels, s=20)
plt.title('SVM决策边界')
plt.show()