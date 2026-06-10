# H2O AutoML学习
# 主要内容：初始化、数据加载、AutoML训练、模型评估

import h2o
from h2o.automl import H2OAutoML

print("=== 初始化H2O ===")
h2o.init()

print("\n=== 创建示例数据 ===")
import pandas as pd
import numpy as np

n = 1000
X = np.random.randn(n, 5)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

df = pd.DataFrame(X, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'])
df['target'] = y

h2o_df = h2o.H2OFrame(df)
print(f"数据形状: {h2o_df.shape}")
print(f"数据前5行:\n{h2o_df.head()}")

print("\n=== 划分训练集和测试集 ===")
train, test = h2o_df.split_frame(ratios=[0.8], seed=42)
print(f"训练集大小: {train.shape}")
print(f"测试集大小: {test.shape}")

print("\n=== 定义特征和目标 ===")
x = train.columns[:-1]
y = 'target'

print("\n=== 创建AutoML ===")
aml = H2OAutoML(max_models=5, seed=42)

print("\n=== 训练模型 ===")
aml.train(x=x, y=y, training_frame=train)

print("\n=== 查看排行榜 ===")
lb = aml.leaderboard
print(f"排行榜:\n{lb}")

print("\n=== 获取最佳模型 ===")
best_model = aml.leader
print(f"最佳模型: {best_model}")

print("\n=== 评估模型 ===")
perf = best_model.model_performance(test)
print(f"模型性能:\n{perf}")

print("\n=== 预测 ===")
predictions = best_model.predict(test)
print(f"预测结果前5行:\n{predictions.head()}")

print("\n=== 关闭H2O ===")
h2o.shutdown(prompt=False)