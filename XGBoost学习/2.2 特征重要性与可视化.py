# XGBoost特征重要性与可视化学习
# 主要内容：特征重要性分析、SHAP值、模型可视化

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("=== 加载数据集 ===")
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n=== 训练XGBoost模型 ===")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'multi:softmax',
    'num_class': 3,
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss'
}

model = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, 'test')])

print("\n=== 特征重要性 ===")
importance = model.get_score(importance_type='weight')
print("特征重要性(weight):")
for feature, score in importance.items():
    print(f"  {feature}: {score}")

importance_gain = model.get_score(importance_type='gain')
print("\n特征重要性(gain):")
for feature, score in importance_gain.items():
    print(f"  {feature}: {score:.4f}")

print("\n=== 特征重要性可视化 ===")
xgb.plot_importance(model)
plt.title('XGBoost特征重要性')
plt.show()

print("\n=== 树结构可视化 ===")
xgb.plot_tree(model, num_trees=0)
plt.show()

print("\n=== 模型预测 ===")
y_pred = model.predict(dtest)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")

print("\n=== SHAP值分析 ===")
try:
    import shap
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    
    shap.summary_plot(shap_values, X_train, feature_names=iris.feature_names)
    print("SHAP摘要图已生成")
    
    shap.dependence_plot(0, shap_values[0], X_train, feature_names=iris.feature_names)
    print("SHAP依赖图已生成")
except ImportError:
    print("SHAP未安装，跳过SHAP分析")

print("\n=== 保存模型 ===")
model.save_model('xgb_model.json')
print("模型已保存")

print("\n=== 加载模型 ===")
loaded_model = xgb.Booster()
loaded_model.load_model('xgb_model.json')
print("模型已加载")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('xgb_model.json'):
    os.remove('xgb_model.json')
    print("已删除测试模型")