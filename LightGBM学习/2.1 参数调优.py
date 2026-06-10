# LightGBM参数调优学习
# 主要内容：参数调整、早停机制、交叉验证、特征重要性

import lightgbm as lgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n=== 创建Dataset ===")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

print("\n=== 基础参数 ===")
params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5
}

print("\n=== 训练模型 ===")
model = lgb.train(params, train_data, num_boost_round=100, valid_sets=[test_data])

print("\n=== 早停机制 ===")
early_stopping = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[test_data],
    early_stopping_rounds=10
)
print(f"早停轮数: {early_stopping.best_iteration}")

print("\n=== 参数调优 ===")
param_grid = {
    'num_leaves': [15, 31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.8, 0.9, 1.0]
}

estimator = lgb.LGBMClassifier(objective='binary')
grid_search = GridSearchCV(estimator, param_grid, cv=3)
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳得分: {grid_search.best_score_:.4f}")

print("\n=== 特征重要性 ===")
model = lgb.LGBMClassifier(**grid_search.best_params_)
model.fit(X_train, y_train)

lgb.plot_importance(model)
plt.title('LightGBM特征重要性')
plt.show()

print("\n=== 树结构可视化 ===")
lgb.plot_tree(model, tree_index=0)
plt.show()

print("\n=== 模型评估 ===")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

print("\n=== 保存模型 ===")
model.booster_.save_model('lgb_model.txt')
print("模型已保存")