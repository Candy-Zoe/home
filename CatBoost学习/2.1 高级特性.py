# CatBoost高级特性学习
# 主要内容：处理类别特征、文本特征、GPU训练、模型解释

from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("=== 创建示例数据 ===")
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(f"训练集形状: {X_train.shape}")

print("\n=== 基础分类 ===")
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=10
)

model.fit(X_train, y_train, eval_set=(X_test, y_test))
print("模型训练完成")

print("\n=== 处理类别特征 ===")
df = pd.DataFrame({
    'feature1': np.random.randn(1000),
    'feature2': np.random.randn(1000),
    'category': np.random.choice(['A', 'B', 'C'], 1000),
    'target': np.random.randint(0, 2, 1000)
})

X = df.drop('target', axis=1)
y = df['target']
cat_features = ['category']

model_cat = CatBoostClassifier(iterations=100, verbose=10)
model_cat.fit(X, y, cat_features=cat_features)
print("类别特征处理完成")

print("\n=== Grid Search ===")
from sklearn.model_selection import GridSearchCV

param_grid = {
    'iterations': [50, 100],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1]
}

model_grid = CatBoostClassifier(verbose=0)
grid_search = GridSearchCV(model_grid, param_grid, cv=3)
grid_search.fit(X_train, y_train)
print(f"最佳参数: {grid_search.best_params_}")

print("\n=== 特征重要性 ===")
feature_importance = model.get_feature_importance()
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]

plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance)
plt.title('CatBoost特征重要性')
plt.xlabel('重要性')
plt.tight_layout()
plt.show()

print("\n=== SHAP值分析 ===")
try:
    from shap import TreeExplainer, summary_plot
    
    explainer = TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    summary_plot(shap_values, X_test, feature_names=feature_names)
    print("SHAP摘要图已生成")
except ImportError:
    print("SHAP未安装，跳过SHAP分析")

print("\n=== 回归任务 ===")
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg)

regressor = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=10
)

regressor.fit(X_train_reg, y_train_reg)
print("回归模型训练完成")

print("\n=== 回归评估 ===")
from sklearn.metrics import mean_squared_error, r2_score

predictions = regressor.predict(X_test_reg)
mse = mean_squared_error(y_test_reg, predictions)
r2 = r2_score(y_test_reg, predictions)

print(f"MSE: {mse:.4f}")
print(f"R2: {r2:.4f}")

print("\n=== 早停机制 ===")
model_early = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    early_stopping_rounds=50,
    verbose=100
)

model_early.fit(X_train, y_train, eval_set=(X_test, y_test))
print(f"最佳迭代次数: {model_early.get_best_iteration()}")

print("\n=== 模型保存和加载 ===")
model.save_model('catboost_model.cbm')
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')
print("模型已保存和加载")

print("\n=== Pool对象 ===")
train_pool = Pool(X_train, y_train, cat_features=cat_features if 'cat_features' in dir() else None)
test_pool = Pool(X_test, y_test)

model_pool = CatBoostClassifier(iterations=100, verbose=10)
model_pool.fit(train_pool, eval_set=test_pool)
print("Pool对象训练完成")

print("\n=== 清理 ===")
import os
if os.path.exists('catboost_model.cbm'):
    os.remove('catboost_model.cbm')
    print("模型文件已删除")