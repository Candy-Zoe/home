# CatBoost回归任务学习
# 主要内容：使用CatBoost进行回归任务

import catboost as cb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=== 加载数据集 ===")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 创建Pool ===")
train_pool = cb.Pool(X_train, y_train)
test_pool = cb.Pool(X_test, y_test)

print("\n=== 设置参数 ===")
params = {
    'iterations': 100,
    'learning_rate': 0.05,
    'depth': 3,
    'loss_function': 'RMSE'
}

print("\n=== 训练模型 ===")
model = cb.CatBoostRegressor(**params)
model.fit(train_pool, eval_set=test_pool, verbose=10)

print("\n=== 预测 ===")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

print("\n=== 使用scikit-learn接口 ===")
reg = cb.CatBoostRegressor(iterations=100, learning_rate=0.05, depth=3)
reg.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=0)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")