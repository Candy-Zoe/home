# CatBoost分类与回归基础学习
# 主要内容：CatBoost分类器、回归器的使用、类别特征处理

# 导入必要的库
import catboost as cb
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

# 分类任务示例
print("=== CatBoost分类示例 ===")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建CatBoost分类器
clf = cb.CatBoostClassifier(
    iterations=100,        # 迭代次数
    depth=3,               # 树的深度
    learning_rate=0.1,     # 学习率
    loss_function='MultiClass',  # 多分类损失函数
    random_state=42,
    verbose=False          # 关闭训练日志
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
print("\n=== CatBoost回归示例 ===")

# 加载波士顿房价数据集
boston = load_boston()
X = boston.data
y = boston.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建CatBoost回归器
reg = cb.CatBoostRegressor(
    iterations=100,        # 迭代次数
    depth=3,               # 树的深度
    learning_rate=0.1,     # 学习率
    loss_function='RMSE',  # 回归损失函数
    random_state=42,
    verbose=False          # 关闭训练日志
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

# 类别特征处理示例
print("\n=== 类别特征处理 ===")

# 创建包含类别特征的示例数据
data = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [5000, 8000, 6000, 10000, 7000],
    'education': ['本科', '硕士', '本科', '博士', '硕士'],
    'city': ['北京', '上海', '北京', '深圳', '上海'],
    'label': [0, 1, 0, 1, 0]
})

print("原始数据:")
print(data)

# 分离特征和标签
X = data.drop('label', axis=1)
y = data['label']

# 指定类别特征索引
cat_features = ['education', 'city']

# 创建分类器并训练
clf_cat = cb.CatBoostClassifier(
    iterations=50,
    depth=2,
    learning_rate=0.1,
    cat_features=cat_features,  # 指定类别特征
    random_state=42,
    verbose=False
)

clf_cat.fit(X, y)
print("\n类别特征模型训练完成")

# 预测
y_pred = clf_cat.predict(X)
print(f"预测结果: {y_pred}")

# 特征重要性
print("\n=== 特征重要性 ===")

# 获取特征重要性
feature_importance = reg.get_feature_importance()

# 创建特征重要性DataFrame
importance_df = pd.DataFrame({
    '特征': boston.feature_names,
    '重要性': feature_importance
}).sort_values(by='重要性', ascending=False)

print("特征重要性排名:")
print(importance_df)

# 自定义损失函数示例
print("\n=== 自定义评估指标 ===")

# 使用自定义评估指标训练
reg_custom = cb.CatBoostRegressor(
    iterations=100,
    eval_metric='MAE',  # 使用MAE作为评估指标
    random_state=42,
    verbose=False
)

reg_custom.fit(X_train, y_train, eval_set=(X_test, y_test))

# 获取最佳分数
best_score = reg_custom.get_best_score()
print(f"最佳MAE分数: {best_score['validation']['MAE']:.4f}")