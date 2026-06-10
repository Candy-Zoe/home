# LightGBM回归任务学习
# 主要内容：使用LightGBM进行回归任务

import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("=== 加载数据集 ===")
boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n=== 创建Dataset ===")
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_test = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

print("\n=== 设置参数 ===")
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05
}

print("\n=== 训练模型 ===")
model = lgb.train(params, lgb_train, num_boost_round=100, valid_sets=[lgb_test])

print("\n=== 预测 ===")
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

print("\n=== 使用scikit-learn接口 ===")
from lightgbm import LGBMRegressor

reg = LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.05)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")