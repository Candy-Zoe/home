# H2O自动化机器学习进阶学习
# 主要内容：H2O深度学习、H2O AutoML进阶、模型解释、特征工程

import h2o
from h2o.automl import H2OAutoML
from h2o.grid.grid_search import H2OGridSearch
from h2o.tree import H2OTree
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

h2o.init()

print("=== 创建示例数据 ===")
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'feature1': np.random.randn(n),
    'feature2': np.random.randn(n),
    'feature3': np.random.randn(n),
    'target': (np.random.randn(n) > 0).astype(int)
})

hf = h2o.H2OFrame(df)
hf['target'] = hf['target'].asfactor()

print(f"数据形状: {hf.shape}")

print("\n=== H2O AutoML ===")
aml = H2OAutoML(
    max_models=10,
    max_runtime_secs=60,
    seed=42,
    balance_classes=True
)

train, test = hf.split_frame(ratios=[0.8], seed=42)

aml.train(y='target', training_frame=train, validation_frame=test)
print("AutoML训练完成")

print("\n=== 查看Leaderboard ===")
lb = aml.leaderboard
print(lb.head())

print("\n=== 获取最佳模型 ===")
best_model = aml.leader
print(f"最佳模型: {best_model.model_id}")

print("\n=== 模型性能评估 ===")
performance = best_model.model_performance(test)
print(performance)

print("\n=== 特征重要性 ===")
if hasattr(best_model, 'varimp'):
    varimp = best_model.varimp()
    print("特征重要性:")
    for var, imp in varimp[:5]:
        print(f"  {var}: {imp:.4f}")

print("\n=== H2O深度学习 ===")
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

dl_model = H2ODeepLearningEstimator(
    hidden=[50, 50],
    epochs=10,
    activation='Rectifier',
    train_samples_per_iteration=-1,
    reproducible=True,
    seed=42
)

dl_model.train(y='target', training_frame=train, validation_frame=test)
print("深度学习模型训练完成")

print("\n=== 模型预测 ===")
predictions = best_model.predict(test)
print(f"预测结果前5行:\n{predictions.head()}")

print("\n=== 部分依赖图 ===")
pd_plot = best_model.partial_plot(train, cols=['feature1'], plot=True)
print("部分依赖图已生成")

print("\n=== SHAP解释 ===")
try:
    explainer = best_model.explain(test)
    print("模型解释已生成")
except:
    print("SHAP解释需要额外配置")

print("\n=== 模型保存和加载 ===")
model_path = h2o.save_model(model=best_model, path="./h2o_models", force=True)
print(f"模型已保存到: {model_path}")

loaded_model = h2o.load_model(model_path)
print("模型已加载")

print("\n=== 网格搜索 ===")
from h2o.estimators.gbm import H2OGradientBoostingEstimator

gbm_params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

gbm_grid = H2OGridSearch(
    model=H2OGradientBoostingEstimator(ntrees=100, seed=42),
    hyper_params=gbm_params
)

gbm_grid.train(y='target', training_frame=train, validation_frame=test)
print("网格搜索完成")

best_gbm = gbm_grid.get_grid(sort_by='auc', decreasing=True).models[0]
print(f"最佳GBM参数: {best_gbm.params}")

print("\n=== 关闭H2O ===")
h2o.cluster().shutdown()
print("H2O集群已关闭")

print("\n=== 清理测试文件 ===")
import shutil
import os
for path in ["./h2o_models"]:
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"已删除 {path}")