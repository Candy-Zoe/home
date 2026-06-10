# scikit-learn模型选择与评估进阶学习
# 主要内容：交叉验证策略、性能指标、模型持久化、管道进阶

from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import (
    cross_val_score, KFold, StratifiedKFold, 
    LeaveOneOut, ShuffleSplit, GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import numpy as np
import matplotlib.pyplot as plt

print("=== 加载数据 ===")
data = load_breast_cancer()
X, y = data.data, data.target
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

print("\n=== 交叉验证策略 ===")
model = LogisticRegression(max_iter=1000)

kfold = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
print(f"KFold准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(model, X, y, cv=stratified, scoring='accuracy')
print(f"分层KFold准确率: {cv_scores_stratified.mean():.4f}")

loo = LeaveOneOut()
cv_scores_loo = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
print(f"留一法准确率: {cv_scores_loo.mean():.4f}")

print("\n=== GridSearchCV ===")
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}

from sklearn.svm import SVC
grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.4f}")

print("\n=== RandomizedSearchCV ===")
from scipy.stats import uniform, randint

param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(
    RandomForestClassifier(), param_dist, 
    n_iter=20, cv=5, scoring='accuracy', random_state=42
)
random_search.fit(X, y)
print(f"随机搜索最佳得分: {random_search.best_score_:.4f}")

print("\n=== 性能指标 ===")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"精确率: {precision_score(y_test, y_pred):.4f}")
print(f"召回率: {recall_score(y_test, y_pred):.4f}")
print(f"F1分数: {f1_score(y_test, y_pred):.4f}")

print("\n=== 混淆矩阵 ===")
cm = confusion_matrix(y_test, y_pred)
print(f"混淆矩阵:\n{cm}")

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

print("\n=== ROC曲线 ===")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend()
plt.show()

print("\n=== Precision-Recall曲线 ===")
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

plt.figure()
plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {avg_precision:.2f})')
plt.xlabel('召回率')
plt.ylabel('精确率')
plt.title('Precision-Recall曲线')
plt.legend()
plt.show()

print("\n=== 交叉验证可视化 ===")
from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(), X, y, 
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring='accuracy'
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, color='blue', label='训练得分')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean, color='green', label='验证得分')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.title('学习曲线')
plt.legend()
plt.show()