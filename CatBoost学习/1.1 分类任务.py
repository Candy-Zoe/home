# CatBoost分类任务学习
# 主要内容：使用CatBoost进行分类任务，自动处理类别特征

import catboost as cb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 创建Pool ===")
train_pool = cb.Pool(X_train, y_train)
test_pool = cb.Pool(X_test, y_test)

print("\n=== 设置参数 ===")
params = {
    'iterations': 100,
    'learning_rate': 0.05,
    'depth': 3,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy'
}

print("\n=== 训练模型 ===")
model = cb.CatBoostClassifier(**params)
model.fit(train_pool, eval_set=test_pool, verbose=10)

print("\n=== 预测 ===")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print("\n=== 特征重要性 ===")
import matplotlib.pyplot as plt
feature_importance = model.get_feature_importance()
plt.bar(range(len(feature_importance)), feature_importance)
plt.title('特征重要性')
plt.show()

print("\n=== 使用scikit-learn接口 ===")
clf = cb.CatBoostClassifier(iterations=100, learning_rate=0.05, depth=3)
clf.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")