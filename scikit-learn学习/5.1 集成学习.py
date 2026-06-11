# scikit-learn集成学习方法学习
# 主要内容：Bagging、Boosting、Voting、Stacking

# 导入必要的库
from sklearn.ensemble import (
    BaggingClassifier, RandomForestClassifier, AdaBoostClassifier,
    GradientBoostingClassifier, VotingClassifier, StackingClassifier,
    BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt

# 生成分类数据
print("=== 生成分类数据 ===")

X, y = make_classification(
    n_samples=500,      # 样本数量
    n_features=20,      # 特征数量
    n_informative=15,   # 有效特征数量
    n_redundant=5,      # 冗余特征数量
    random_state=42
)

print(f"特征矩阵形状: {X.shape}")
print(f"目标向量形状: {y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Bagging
print("\n=== Bagging（套袋法） ===")

# 使用BaggingClassifier包装决策树
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=5),
    n_estimators=10,      # 基学习器数量
    max_samples=0.8,      # 每个基学习器使用的样本比例
    max_features=0.8,     # 每个基学习器使用的特征比例
    random_state=42
)

# 训练
bagging_clf.fit(X_train, y_train)

# 预测
y_pred_bagging = bagging_clf.predict(X_test)
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

print(f"Bagging准确率: {accuracy_bagging:.4f}")

# 单棵决策树对比
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train, y_train)
accuracy_dt = accuracy_score(y_test, dt_clf.predict(X_test))
print(f"单棵决策树准确率: {accuracy_dt:.4f}")

# 随机森林
print("\n=== 随机森林 ===")

rf_clf = RandomForestClassifier(
    n_estimators=100,     # 树的数量
    max_depth=10,        # 最大深度
    min_samples_split=5,  # 分裂最小样本数
    random_state=42,
    n_jobs=-1            # 使用所有CPU核心
)

# 训练
rf_clf.fit(X_train, y_train)

# 预测
y_pred_rf = rf_clf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)

print(f"随机森林准确率: {accuracy_rf:.4f}")

# 特征重要性
feature_importance = rf_clf.feature_importances_
top_10_idx = np.argsort(feature_importance)[::-1][:10]

print("\n特征重要性排名（Top 10）:")
for i, idx in enumerate(top_10_idx, 1):
    print(f"  {i}. 特征{idx}: {feature_importance[idx]:.4f}")

# AdaBoost
print("\n=== AdaBoost（自适应提升） ===")

ada_clf = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # 弱学习器
    n_estimators=50,      # 迭代次数
    learning_rate=1.0,    # 学习率
    random_state=42
)

# 训练
ada_clf.fit(X_train, y_train)

# 预测
y_pred_ada = ada_clf.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred_ada)

print(f"AdaBoost准确率: {accuracy_ada:.4f}")

# Gradient Boosting
print("\n=== Gradient Boosting（梯度提升） ===")

gb_clf = GradientBoostingClassifier(
    n_estimators=100,     # 树的数量
    max_depth=5,         # 最大深度
    learning_rate=0.1,    # 学习率
    subsample=0.8,       # 子采样比例
    random_state=42
)

# 训练
gb_clf.fit(X_train, y_train)

# 预测
y_pred_gb = gb_clf.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)

print(f"Gradient Boosting准确率: {accuracy_gb:.4f}")

# Voting（投票法）
print("\n=== Voting（投票法） ===")

# 创建多个不同的分类器
clf1 = LogisticRegression(max_iter=200, random_state=42)
clf2 = DecisionTreeClassifier(max_depth=5, random_state=42)
clf3 = SVC(kernel='rbf', probability=True, random_state=42)

# 硬投票（多数表决）
voting_hard = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
    voting='hard'
)

# 软投票（概率加权）
voting_soft = VotingClassifier(
    estimators=[('lr', clf1), ('dt', clf2), ('svc', clf3)],
    voting='soft'
)

# 训练
voting_hard.fit(X_train, y_train)
voting_soft.fit(X_train, y_train)

# 预测和评估
y_pred_voting_hard = voting_hard.predict(X_test)
y_pred_voting_soft = voting_soft.predict(X_test)

accuracy_voting_hard = accuracy_score(y_test, y_pred_voting_hard)
accuracy_voting_soft = accuracy_score(y_test, y_pred_voting_soft)

print(f"Voting（硬投票）准确率: {accuracy_voting_hard:.4f}")
print(f"Voting（软投票）准确率: {accuracy_voting_soft:.4f}")

# Stacking（堆叠法）
print("\n=== Stacking（堆叠法） ===")

# 创建Stacking分类器
# 基学习器
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42))
]

# 元学习器
stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=200),
    cv=5  # 交叉验证折数
)

# 训练
stacking_clf.fit(X_train, y_train)

# 预测
y_pred_stacking = stacking_clf.predict(X_test)
accuracy_stacking = accuracy_score(y_test, y_pred_stacking)

print(f"Stacking准确率: {accuracy_stacking:.4f}")

# 模型对比
print("\n=== 模型对比 ===")

models = {
    '单棵决策树': accuracy_dt,
    'Bagging': accuracy_bagging,
    '随机森林': accuracy_rf,
    'AdaBoost': accuracy_ada,
    'Gradient Boosting': accuracy_gb,
    'Voting（硬投票）': accuracy_voting_hard,
    'Voting（软投票）': accuracy_voting_soft,
    'Stacking': accuracy_stacking
}

print("各模型准确率对比:")
for name, acc in sorted(models.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {acc:.4f}")

# 可视化对比
plt.figure(figsize=(10, 6))
names = list(models.keys())
accuracies = list(models.values())

bars = plt.barh(names, accuracies)
plt.xlabel('准确率')
plt.title('集成学习算法对比')
plt.xlim(0.5, 1.0)

# 在条形图上添加数值标签
for bar, acc in zip(bars, accuracies):
    plt.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center')

plt.tight_layout()
plt.show()

# 回归任务示例
print("\n=== 回归任务示例 ===")

# 生成回归数据
X_reg, y_reg = make_regression(
    n_samples=500,
    n_features=10,
    noise=10,
    random_state=42
)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

# 随机森林回归
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)
y_pred_rf_reg = rf_reg.predict(X_test_reg)
mse_rf = mean_squared_error(y_test_reg, y_pred_rf_reg)
print(f"随机森林回归 MSE: {mse_rf:.4f}")

# Gradient Boosting回归
gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_reg.fit(X_train_reg, y_train_reg)
y_pred_gb_reg = gb_reg.predict(X_test_reg)
mse_gb = mean_squared_error(y_test_reg, y_pred_gb_reg)
print(f"Gradient Boosting回归 MSE: {mse_gb:.4f}")

# 交叉验证对比
print("\n=== 交叉验证对比 ===")

classifiers = {
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42)
}

cv_results = []
for name, clf in classifiers.items():
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    cv_results.append({
        '模型': name,
        '平均准确率': scores.mean(),
        '标准差': scores.std()
    })
    print(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

cv_results_df = pd.DataFrame(cv_results)
print("\n交叉验证结果:")
print(cv_results_df)

# 学习曲线
print("\n=== 学习曲线 ===")

from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, cv=5):
    """绘制学习曲线"""
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    return train_sizes, train_mean, train_std, test_mean, test_std

# 绘制学习曲线对比
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, (name, clf) in zip(axes, classifiers.items()):
    train_sizes, train_mean, train_std, test_mean, test_std = plot_learning_curve(
        clf, name, X, y
    )

    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='orange')
    ax.plot(train_sizes, train_mean, 'o-', color='blue', label='训练分数')
    ax.plot(train_sizes, test_mean, 'o-', color='orange', label='验证分数')
    ax.set_title(f'{name}学习曲线')
    ax.set_xlabel('训练样本数')
    ax.set_ylabel('准确率')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n集成学习学习完成！")