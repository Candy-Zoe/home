# XGBoost分类任务学习
# 主要内容：使用XGBoost进行分类任务

import xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 创建DMatrix ===")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("\n=== 设置参数 ===")
params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'eta': 0.1,
    'eval_metric': 'mlogloss'
}

print("\n=== 训练模型 ===")
model = xgb.train(params, dtrain, num_boost_round=100)

print("\n=== 预测 ===")
y_pred = model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print("\n=== 特征重要性 ===")
import matplotlib.pyplot as plt
xgb.plot_importance(model)
plt.title('特征重要性')
plt.show()

print("\n=== 使用scikit-learn接口 ===")
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")