# LightGBM分类与回归基础学习
# 主要内容：LightGBM分类器、回归器的使用、参数设置

# 导入必要的库
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 分类任务示例
print("=== LightGBM分类示例 ===")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建LightGBM分类器
clf = lgb.LGBMClassifier(
    n_estimators=100,      # 树的数量
    max_depth=3,           # 树的最大深度
    learning_rate=0.1,     # 学习率
    objective='multiclass',  # 多分类目标函数
    num_class=3,           # 类别数量
    random_state=42
)

# 训练模型
clf.fit(X_train, y_train)
print("分类模型训练完成")

# 进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")

# 回归任务示例
print("\n=== LightGBM回归示例 ===")

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建LightGBM回归器
reg = lgb.LGBMRegressor(
    n_estimators=100,      # 树的数量
    max_depth=3,           # 树的最大深度
    learning_rate=0.1,     # 学习率
    objective='regression',  # 回归目标函数
    random_state=42
)

# 训练模型
reg.fit(X_train, y_train)
print("回归模型训练完成")

# 进行预测
y_pred = reg.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"测试集均方误差: {mse:.4f}")
print(f"测试集均方根误差: {rmse:.4f}")

# 使用Dataset（LightGBM原生数据结构）
print("\n=== 使用Dataset ===")

# 创建Dataset
lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# 设置参数
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# 训练模型
model = lgb.train(params, lgb_train,
                  num_boost_round=100,
                  valid_sets=lgb_eval,
                  early_stopping_rounds=10)

# 预测
y_pred = model.predict(X_test, num_iteration=model.best_iteration)

# 评估
mse = mean_squared_error(y_test, y_pred)
print(f"Dataset训练 - 均方误差: {mse:.4f}")

# 特征重要性
print("\n=== 特征重要性 ===")

# 获取特征重要性
feature_importance = reg.feature_importances_

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    '特征': boston.feature_names,
    '重要性': feature_importance
}).sort_values(by='重要性', ascending=False)

print("特征重要性排名:")
print(importance_df)

# 交叉验证示例
print("\n=== 交叉验证 ===")

# 执行交叉验证
cv_results = lgb.cv(
    params,
    lgb_train,
    num_boost_round=100,
    nfold=5,
    metrics='rmse',
    early_stopping_rounds=10,
    stratified=False
)

print(f"交叉验证完成")
print(f"最佳迭代轮数: {len(cv_results['rmse-mean'])}")
print(f"交叉验证RMSE均值: {cv_results['rmse-mean'][-1]:.4f}")
print(f"交叉验证RMSE标准差: {cv_results['rmse-stdv'][-1]:.4f}")