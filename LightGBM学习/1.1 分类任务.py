# LightGBM分类任务学习
# 主要内容：使用LightGBM进行分类任务

import lightgbm as lgb
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 创建Dataset ===")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

print("\n=== 设置参数 ===")
params = {
    'objective': 'multiclass',
    'num_class': 3,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05
}

print("\n=== 训练模型 ===")
model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_test])

print("\n=== 预测 ===")
y_pred = model.predict(X_test).argmax(axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print("\n=== 特征重要性 ===")
import matplotlib.pyplot as plt
lgb.plot_importance(model)
plt.title('特征重要性')
plt.show()

print("\n=== 使用scikit-learn接口 ===")
from lightgbm import LGBMClassifier

clf = LGBMClassifier(n_estimators=100, num_leaves=31, learning_rate=0.05)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")