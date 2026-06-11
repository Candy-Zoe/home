# scikit-learn线性回归学习
# 主要内容：简单线性回归、多元线性回归、岭回归、Lasso回归

# 导入必要的库
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# 生成示例数据
print("=== 生成示例数据 ===")

# 生成简单的回归数据
X, y = make_regression(
    n_samples=100,     # 样本数量
    n_features=1,      # 特征数量
    noise=10,          # 噪声标准差
    random_state=42
)

print(f"特征矩阵形状: {X.shape}")
print(f"目标向量形状: {y.shape}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"训练集大小: {len(X_train)}")
print(f"测试集大小: {len(X_test)}")

# 简单线性回归
print("\n=== 简单线性回归 ===")

# 创建线性回归模型
lr = LinearRegression()

# 训练模型
lr.fit(X_train, y_train)
print("线性回归模型训练完成")

# 查看模型参数
print(f"斜率（系数）: {lr.coef_[0]:.4f}")
print(f"截距: {lr.intercept_:.4f}")
print(f"回归方程: y = {lr.coef_[0]:.4f} * x + {lr.intercept_:.4f}")

# 预测
y_pred = lr.predict(X_test)

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n评估指标:")
print(f"  均方误差 (MSE): {mse:.4f}")
print(f"  均方根误差 (RMSE): {rmse:.4f}")
print(f"  平均绝对误差 (MAE): {mae:.4f}")
print(f"  R²分数: {r2:.4f}")

# 可视化回归结果
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, alpha=0.7, label='实际值')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='预测值')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归拟合')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.7)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 多元线性回归
print("\n=== 多元线性回归 ===")

# 加载糖尿病数据集
diabetes = load_diabetes()
X_multi = diabetes.data
y_multi = diabetes.target

print(f"特征数量: {X_multi.shape[1]}")
print(f"特征名称: {diabetes.feature_names}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.3, random_state=42
)

# 训练模型
lr_multi = LinearRegression()
lr_multi.fit(X_train, y_train)

# 预测
y_pred = lr_multi.predict(X_test)

# 评估
r2 = r2_score(y_test, y_pred)
print(f"R²分数: {r2:.4f}")

# 交叉验证
cv_scores = cross_val_score(LinearRegression(), X_multi, y_multi, cv=5, scoring='r2')
print(f"\n5折交叉验证 R²分数: {cv_scores}")
print(f"平均 R²分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 岭回归（L2正则化）
print("\n=== 岭回归 (L2正则化) ===")

# 创建岭回归模型
ridge = Ridge(alpha=1.0)  # alpha是正则化强度

# 训练
ridge.fit(X_train, y_train)

# 预测
y_pred_ridge = ridge.predict(X_test)

# 评估
r2_ridge = r2_score(y_test, y_pred_ridge)
print(f"岭回归 R²分数: {r2_ridge:.4f}")

# 对比不同alpha值的效果
alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
ridge_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    score = ridge.score(X_test, y_test)
    ridge_scores.append(score)
    print(f"  alpha={alpha:6.3f}: R²={score:.4f}")

# 绘制alpha与R²的关系
plt.figure(figsize=(8, 4))
plt.plot(alphas, ridge_scores, marker='o')
plt.xscale('log')
plt.xlabel('Alpha (对数尺度)')
plt.ylabel('R²分数')
plt.title('岭回归：正则化参数对模型性能的影响')
plt.grid(True, alpha=0.3)
plt.show()

# Lasso回归（L1正则化）
print("\n=== Lasso回归 (L1正则化) ===")

# 创建Lasso回归模型
lasso = Lasso(alpha=0.1)

# 训练
lasso.fit(X_train, y_train)

# 预测
y_pred_lasso = lasso.predict(X_test)

# 评估
r2_lasso = r2_score(y_test, y_pred_lasso)
print(f"Lasso回归 R²分数: {r2_lasso:.4f}")

# 查看特征稀疏性（有多少系数被压缩为0）
n_nonzero = np.sum(lasso.coef_ != 0)
print(f"非零系数数量: {n_nonzero}/{len(lasso.coef_)}")

# ElasticNet回归（L1+L2正则化）
print("\n=== ElasticNet回归 (L1+L2正则化) ===")

# 创建ElasticNet模型
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio控制L1和L2的混合比例

# 训练
elastic.fit(X_train, y_train)

# 预测
y_pred_elastic = elastic.predict(X_test)

# 评估
r2_elastic = r2_score(y_test, y_pred_elastic)
print(f"ElasticNet回归 R²分数: {r2_elastic:.4f}")

# 模型对比
print("\n=== 模型对比 ===")

models = {
    '线性回归': lr_multi,
    '岭回归': ridge,
    'Lasso回归': lasso,
    'ElasticNet': elastic
}

results = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        '模型': name,
        'R²': r2,
        'RMSE': rmse
    })

results_df = pd.DataFrame(results)
print(results_df)

# 残差分析
print("\n=== 残差分析 ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, model) in zip(axes.flat, models.items()):
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(y=0, color='red', linestyle='--')
    ax.set_xlabel('预测值')
    ax.set_ylabel('残差')
    ax.set_title(f'{name} - 残差图')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 预测新数据
print("\n=== 预测新数据 ===")

# 创建新数据点
X_new = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# 使用线性回归预测
prediction = lr_multi.predict(X_new)
print(f"新数据预测结果: {prediction[0]:.4f}")