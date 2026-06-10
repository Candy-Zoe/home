# LightGBM高级特性学习
# 主要内容：类别特征处理、自定义损失函数、模型融合、特征工程

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

print("=== 加载数据 ===")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

print("\n=== 创建Dataset ===")
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

print("\n=== 类别特征处理 ===")
df = pd.DataFrame({
    'num1': np.random.randn(1000),
    'num2': np.random.randn(1000),
    'cat1': np.random.choice(['A', 'B', 'C'], 1000),
    'cat2': np.random.choice(['X', 'Y', 'Z'], 1000),
    'target': np.random.randint(0, 2, 1000)
})

df['cat1'] = df['cat1'].astype('category')
df['cat2'] = df['cat2'].astype('category')

print("类别特征已转换")
print(f"数据类型:\n{df.dtypes}")

print("\n=== 自定义损失函数 ===")
def custom_loss(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    grad = y_pred - y_true
    hess = y_pred * (1.0 - y_pred)
    return grad, hess

def custom_eval(y_true, y_pred):
    y_pred = 1.0 / (1.0 + np.exp(-y_pred))
    return 'custom_auc', roc_auc_score(y_true, y_pred), True

print("自定义损失函数已定义")

print("\n=== 模型训练 ===")
params = {
    'objective': 'binary',
    'metric': ['auc', 'binary_logloss'],
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}

evals_result = {}
model = lgb.train(
    params,
    train_data,
    num_boost_round=100,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'valid'],
    evals_result=evals_result,
    early_stopping_rounds=10,
    verbose_eval=20
)

print(f"\n最佳迭代次数: {model.best_iteration}")

print("\n=== 学习曲线 ===")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(evals_result['train']['auc'], label='Train')
plt.plot(evals_result['valid']['auc'], label='Valid')
plt.xlabel('迭代次数')
plt.ylabel('AUC')
plt.title('AUC曲线')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(evals_result['train']['binary_logloss'], label='Train')
plt.plot(evals_result['valid']['binary_logloss'], label='Valid')
plt.xlabel('迭代次数')
plt.ylabel('Log Loss')
plt.title('损失曲线')
plt.legend()

plt.tight_layout()
plt.show()

print("\n=== 特征重要性 ===")
importance = model.feature_importance(importance_type='gain')
feature_names = model.feature_name()

plt.figure(figsize=(10, 6))
indices = np.argsort(importance)[::-1][:10]
plt.barh(range(len(indices)), importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('重要性')
plt.title('特征重要性 (Top 10)')
plt.tight_layout()
plt.show()

print("\n=== 模型融合 ===")
models = []
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    X_tr, X_val = X_train[train_idx], X_train[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    
    train_data = lgb.Dataset(X_tr, label=y_tr)
    val_data = lgb.Dataset(X_val, label=y_val)
    
    model_fold = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[val_data],
        early_stopping_rounds=10,
        verbose_eval=False
    )
    
    models.append(model_fold)
    print(f"Fold {fold+1} 训练完成")

print(f"\n融合模型数量: {len(models)}")

print("\n=== 融合预测 ===")
predictions = np.zeros(X_test.shape[0])
for model_fold in models:
    predictions += model_fold.predict(X_test) / len(models)

y_pred = (predictions > 0.5).astype(int)
print(f"融合准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"融合AUC: {roc_auc_score(y_test, predictions):.4f}")

print("\n=== 模型保存 ===")
model.save_model('lgb_model.txt')
print("模型已保存")

print("\n=== 清理 ===")
import os
if os.path.exists('lgb_model.txt'):
    os.remove('lgb_model.txt')
    print("模型文件已删除")