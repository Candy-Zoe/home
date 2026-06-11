# scikit-learn分类算法学习
# 主要内容：逻辑回归、K近邻、决策树、朴素贝叶斯、支持向量机

# 导入必要的库
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
print("=== 加载数据集 ===")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"特征矩阵形状: {X.shape}")
print(f"类别分布: {np.bincount(y)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 特征标准化（对于某些算法如SVM和KNN很重要）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 逻辑回归
print("\n=== 1. 逻辑回归 ===")

# 创建逻辑回归模型
lr = LogisticRegression(max_iter=200, random_state=42)

# 训练模型
lr.fit(X_train_scaled, y_train)

# 预测
y_pred_lr = lr.predict(X_test_scaled)

# 评估
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"准确率: {accuracy_lr:.4f}")

# 交叉验证
cv_scores_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5)
print(f"5折交叉验证: {cv_scores_lr.mean():.4f} (+/- {cv_scores_lr.std():.4f})")

# K近邻分类器
print("\n=== 2. K近邻分类器 (KNN) ===")

# 创建KNN模型（k=5）
knn = KNeighborsClassifier(n_neighbors=5)

# 训练模型
knn.fit(X_train_scaled, y_train)

# 预测
y_pred_knn = knn.predict(X_test_scaled)

# 评估
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"准确率: {accuracy_knn:.4f}")

# 测试不同的k值
k_values = range(1, 21)
k_scores = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    score = knn_k.score(X_test_scaled, y_test)
    k_scores.append(score)
    if k % 5 == 0:
        print(f"k={k}: 准确率={score:.4f}")

# 绘制k值与准确率的关系
plt.figure(figsize=(8, 4))
plt.plot(k_values, k_scores, marker='o')
plt.xlabel('K值')
plt.ylabel('准确率')
plt.title('KNN: K值对准确率的影响')
plt.grid(True, alpha=0.3)
plt.show()

# 决策树
print("\n=== 3. 决策树 ===")

# 创建决策树模型
dt = DecisionTreeClassifier(max_depth=5, random_state=42)

# 训练模型
dt.fit(X_train, y_train)

# 预测
y_pred_dt = dt.predict(X_test)

# 评估
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"准确率: {accuracy_dt:.4f}")

# 特征重要性
feature_importance = dt.feature_importances_
print("特征重要性:")
for name, importance in zip(iris.feature_names, feature_importance):
    print(f"  {name}: {importance:.4f}")

# 朴素贝叶斯
print("\n=== 4. 朴素贝叶斯 ===")

# 创建朴素贝叶斯模型
nb = GaussianNB()

# 训练模型
nb.fit(X_train, y_train)

# 预测
y_pred_nb = nb.predict(X_test)

# 评估
accuracy_nb = accuracy_score(y_test, y_pred_nb)
print(f"准确率: {accuracy_nb:.4f}")

# 支持向量机
print("\n=== 5. 支持向量机 (SVM) ===")

# 创建SVM模型
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# 训练模型
svm.fit(X_train_scaled, y_train)

# 预测
y_pred_svm = svm.predict(X_test_scaled)

# 评估
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"准确率: {accuracy_svm:.4f}")

# 测试不同的核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_scores = []

for kernel in kernels:
    svm_k = SVC(kernel=kernel, C=1.0, random_state=42)
    svm_k.fit(X_train_scaled, y_train)
    score = svm_k.score(X_test_scaled, y_test)
    kernel_scores.append(score)
    print(f"  kernel={kernel}: 准确率={score:.4f}")

# 模型对比
print("\n=== 模型对比 ===")

models = {
    '逻辑回归': (lr, X_train_scaled, X_test_scaled),
    'K近邻': (knn, X_train_scaled, X_test_scaled),
    '决策树': (dt, X_train, X_test),
    '朴素贝叶斯': (nb, X_train, X_test),
    '支持向量机': (svm, X_train_scaled, X_test_scaled)
}

results = []
for name, (model, X_tr, X_te) in models.items():
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_test, y_pred)
    results.append({'模型': name, '准确率': acc})

results_df = pd.DataFrame(results)
print(results_df.sort_values('准确率', ascending=False))

# 可视化混淆矩阵
print("\n=== 混淆矩阵可视化 ===")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for ax, (name, (model, X_tr, X_te)) in zip(axes.flat, models.items()):
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_test, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    ax.set_title(f'{name}\n准确率: {accuracy_score(y_test, y_pred):.4f}')
    ax.set_xlabel('预测')
    ax.set_ylabel('实际')

plt.tight_layout()
plt.show()

# 详细分类报告
print("\n=== 详细分类报告 ===")

print("逻辑回归分类报告:")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))

print("\n支持向量机分类报告:")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

# 生成额外的测试数据进行演示
print("\n=== 新数据预测 ===")

# 使用逻辑回归预测新数据
new_data = np.array([[5.1, 3.5, 1.4, 0.2],
                     [6.5, 3.0, 5.5, 2.0],
                     [5.9, 3.0, 5.1, 1.8]])

# 标准化新数据
new_data_scaled = scaler.transform(new_data)

# 预测
predictions = lr.predict(new_data_scaled)
probabilities = lr.predict_proba(new_data_scaled)

print("新数据预测结果:")
for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
    pred_name = iris.target_names[pred]
    print(f"  样本{i+1}: 预测={pred_name}, 概率={prob[pred]:.4f}")