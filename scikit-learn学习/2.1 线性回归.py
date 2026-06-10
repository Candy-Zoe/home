# scikit-learn线性回归学习
# 主要内容：简单线性回归、多元线性回归、正则化回归

import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

print("=== 简单线性回归 ===")
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

model = LinearRegression()
model.fit(X, y)

print(f"系数: {model.coef_[0][0]:.4f}")
print(f"截距: {model.intercept_[0]:.4f}")

y_pred = model.predict(X)

plt.scatter(X, y, label='数据点')
plt.plot(X, y_pred, 'r-', label='拟合直线')
plt.legend()
plt.title('简单线性回归')
plt.show()

print("\n=== 多元线性回归 ===")
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"系数: {model.coef_}")
print(f"截距: {model.intercept_:.4f}")
print(f"R²分数: {r2_score(y_test, y_pred):.4f}")

print("\n=== 正则化回归 - Ridge ===")
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print(f"Ridge R²分数: {r2_score(y_test, y_pred_ridge):.4f}")

print("\n=== 正则化回归 - Lasso ===")
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print(f"Lasso R²分数: {r2_score(y_test, y_pred_lasso):.4f}")
print(f"Lasso系数(稀疏性): {lasso.coef_}")