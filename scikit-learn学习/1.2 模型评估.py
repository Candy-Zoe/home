# scikit-learn模型评估学习
# 主要内容：评估指标、交叉验证、模型选择

# 导入必要的库
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, r2_score
)
import numpy as np
import matplotlib.pyplot as plt

# 分类模型评估
print("=== 分类模型评估 ===")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练模型
model = LogisticRegression(max_iter=200, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")

# 精确率、召回率、F1分数
print("\n=== 多类别分类指标 ===")

# 对每个类别计算指标
precision = precision_score(y_test, y_pred, average=None)
recall = recall_score(y_test, y_pred, average=None)
f1 = f1_score(y_test, y_pred, average=None)

for i in range(len(iris.target_names)):
    print(f"{iris.target_names[i]}: 精确率={precision[i]:.4f}, 召回率={recall[i]:.4f}, F1={f1[i]:.4f}")

# 计算宏平均和加权平均
print(f"\n宏平均 - 精确率: {precision_score(y_test, y_pred, average='macro'):.4f}")
print(f"加权平均 - F1分数: {f1_score(y_test, y_pred, average='weighted'):.4f}")

# 混淆矩阵
print("\n=== 混淆矩阵 ===")

cm = confusion_matrix(y_test, y_pred)
print("混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
tick_marks = np.arange(len(iris.target_names))
plt.xticks(tick_marks, iris.target_names, rotation=45)
plt.yticks(tick_marks, iris.target_names)

# 在每个格子中显示数值
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.show()

# 详细的分类报告
print("\n=== 分类报告 ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 二分类模型评估
print("\n=== 二分类模型评估 ===")

# 加载乳腺癌数据集
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练随机森林模型
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 预测
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"ROC AUC分数: {roc_auc:.4f}")

# 绘制ROC曲线
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (FPR)')
plt.ylabel('真正率 (TPR)')
plt.title('ROC曲线')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()

# 交叉验证
print("\n=== 交叉验证 ===")

# 创建模型
clf = LogisticRegression(max_iter=200, random_state=42)

# 5折交叉验证
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')

print(f"交叉验证分数: {cv_scores}")
print(f"平均准确率: {cv_scores.mean():.4f}")
print(f"标准差: {cv_scores.std():.4f}")

# 使用不同的评分指标
scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']

for metric in scoring_metrics:
    scores = cross_val_score(clf, X, y, cv=5, scoring=metric)
    print(f"{metric}: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 回归模型评估
print("\n=== 回归模型评估 ===")

from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# 加载糖尿病数据集
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 训练线性回归模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_test - y_pred))
r2 = r2_score(y_test, y_pred)

print(f"均方误差 (MSE): {mse:.4f}")
print(f"均方根误差 (RMSE): {rmse:.4f}")
print(f"平均绝对误差 (MAE): {mae:.4f}")
print(f"R²分数: {r2:.4f}")

# 超参数调优
print("\n=== 超参数调优 ===")

# 创建模型
dt = DecisionTreeClassifier(random_state=42)

# 定义参数网格
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 网格搜索
grid_search = GridSearchCV(
    dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

# 使用最佳模型进行预测
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred):.4f}")