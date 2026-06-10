# XGBoost回归任务学习
# 主要内容：使用XGBoost进行回归任务

import xgboost as xgb
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=== 加载数据集 ===")
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 创建DMatrix ===")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

print("\n=== 设置参数 ===")
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'eta': 0.1,
    'eval_metric': 'rmse'
}

print("\n=== 训练模型 ===")
model = xgb.train(params, dtrain, num_boost_round=100)

print("\n=== 预测 ===")
y_pred = model.predict(dtest)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

print("\n=== 使用scikit-learn接口 ===")
from xgboost import XGBRegressor

reg = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")