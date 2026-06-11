# XGBoost分类与回归基础学习
# 主要内容：XGBoost分类器、回归器的使用、参数调优基础

# 导入必要的库
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 分类任务示例
print("=== XGBoost分类示例 ===")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建XGBoost分类器
clf = xgb.XGBClassifier(
    n_estimators=100,      # 树的数量
    max_depth=3,           # 树的最大深度
    learning_rate=0.1,     # 学习率
    objective='multi:softmax',  # 多分类目标函数
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
print("\n=== XGBoost回归示例 ===")

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建XGBoost回归器
reg = xgb.XGBRegressor(
    n_estimators=100,      # 树的数量
    max_depth=3,           # 树的最大深度
    learning_rate=0.1,     # 学习率
    objective='reg:squarederror',  # 回归目标函数
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

# 使用DMatrix（XGBoost原生数据结构）
print("\n=== 使用DMatrix ===")

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置参数
params = {
    'objective': 'reg:squarederror',
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'rmse'
}

# 训练模型
model = xgb.train(params, dtrain, num_boost_round=100)

# 预测
y_pred = model.predict(dtest)

# 评估
mse = mean_squared_error(y_test, y_pred)
print(f"DMatrix训练 - 均方误差: {mse:.4f}")

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

# 早停示例
print("\n=== 早停示例 ===")

# 设置早停参数
reg_early = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.1,
    objective='reg:squarederror',
    early_stopping_rounds=10,  # 早停轮数
    eval_metric='rmse',
    random_state=42
)

# 训练时设置验证集
reg_early.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print(f"早停后实际训练轮数: {reg_early.best_iteration}")
print(f"最佳验证集RMSE: {reg_early.best_score_:.4f}")