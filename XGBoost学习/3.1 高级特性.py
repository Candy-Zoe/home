# XGBoost高级特性学习
# 主要内容：DMatrix优化、自定义目标函数、交叉验证、特征交互

import xgboost as xgb
import numpy as np
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

print("=== 加载数据 ===")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

print("\n=== DMatrix优化 ===")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

dtrain.set_group([X_train.shape[0]])
print("DMatrix已创建并优化")

print("\n=== 自定义目标函数 ===")
def custom_objective(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1.0 - preds)
    return grad, hess

def custom_eval(preds, dtrain):
    labels = dtrain.get_label()
    preds = 1.0 / (1.0 + np.exp(-preds))
    return 'custom_error', np.mean((preds > 0.5) != labels)

print("自定义目标函数已定义")

print("\n=== 交叉验证 ===")
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

cv_results = xgb.cv(
    params,
    dtrain,
    num_boost_round=100,
    nfold=5,
    metrics=['auc', 'error'],
    early_stopping_rounds=10,
    as_pandas=True
)

print(f"交叉验证结果:\n{cv_results.tail()}")
print(f"最佳迭代次数: {cv_results.shape[0]}")

print("\n=== 特征交互 ===")
model = xgb.train(params, dtrain, num_boost_round=100)

interaction_scores = model.get_score(importance_type='gain')
print("特征重要性 (gain):")
for feat, score in sorted(interaction_scores.items(), key=lambda x: -x[1])[:5]:
    print(f"  {feat}: {score:.4f}")

print("\n=== 单特征分析 ===")
from xgboost import plot_importance

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

plot_importance(model, importance_type='weight', ax=axes[0], max_num_features=10)
axes[0].set_title('特征重要性 (weight)')

plot_importance(model, importance_type='gain', ax=axes[1], max_num_features=10)
axes[1].set_title('特征重要性 (gain)')

plt.tight_layout()
plt.show()

print("\n=== 学习曲线 ===")
evals_result = {}
model = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    evals_result=evals_result,
    verbose_eval=False
)

plt.figure(figsize=(10, 5))
plt.plot(evals_result['train']['logloss'], label='Train')
plt.plot(evals_result['test']['logloss'], label='Test')
plt.xlabel('迭代次数')
plt.ylabel('Log Loss')
plt.title('学习曲线')
plt.legend()
plt.show()

print("\n=== 模型快照 ===")
model.save_model('xgb_snapshot.json')
print("模型快照已保存")

print("\n=== 增量训练 ===")
model_partial = xgb.train(params, dtrain, num_boost_round=50)
model_full = xgb.train(params, dtrain, num_boost_round=100, xgb_model=model_partial)
print("增量训练完成")

print("\n=== 特征选择 ===")
from sklearn.feature_selection import SelectFromModel

xgb_clf = xgb.XGBClassifier(**params, n_estimators=100)
xgb_clf.fit(X_train, y_train)

selector = SelectFromModel(xgb_clf, threshold='median', prefit=True)
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

print(f"原始特征数: {X_train.shape[1]}")
print(f"选择后特征数: {X_train_selected.shape[1]}")

print("\n=== 预测概率校准 ===")
y_pred_proba = model.predict(dtest)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\n=== 清理 ===")
import os
if os.path.exists('xgb_snapshot.json'):
    os.remove('xgb_snapshot.json')
    print("模型文件已删除")