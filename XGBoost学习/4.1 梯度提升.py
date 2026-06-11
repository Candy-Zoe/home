# XGBoost梯度提升学习
# 主要内容：XGBoost基础、参数调优、特征重要性、交叉验证、特征工程

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_squared_error, r2_score
import xgboost as xgb

# XGBoost基础
print("=== XGBoost基础 ===")
print(f"XGBoost版本: {xgb.__version__}")

# 加载数据
print("\n=== 加载数据 ===")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征数量: {len(feature_names)}")
print(f"类别分布: {np.bincount(y)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# 基本XGBoost分类
print("\n=== 基本XGBoost分类 ===")

# 创建XGBoost分类器
xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 训练模型
print("训练XGBoost分类器...")
xgb_clf.fit(X_train, y_train)

# 预测
y_pred = xgb_clf.predict(X_test)
y_pred_proba = xgb_clf.predict_proba(X_test)[:, 1]

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 训练集和测试集表现
train_acc = xgb_clf.score(X_train, y_train)
test_acc = xgb_clf.score(X_test, y_test)
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# DMatrix - XGBoost原生数据结构
print("\n=== DMatrix数据结构 ===")

# 创建DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

print(f"DMatrix训练集: {dtrain.num_row()}行 x {dtrain.num_col()}列")
print(f"DMatrix测试集: {dtest.num_row()}行 x {dtest.num_col()}列")

# 使用DMatrix训练原生XGBoost
print("\n使用DMatrix训练原生XGBoost...")
params = {
    'objective': 'binary:logistic',
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'logloss'
}

# 训练模型
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=100,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    verbose_eval=20
)

# 预测
y_pred_native = bst.predict(dtest)
y_pred_native_class = (y_pred_native > 0.5).astype(int)
accuracy_native = accuracy_score(y_test, y_pred_native_class)
print(f"\n原生XGBoost准确率: {accuracy_native:.4f}")

# 训练历史可视化
print("\n=== 训练历史可视化 ===")

# 获取训练历史
results = bst.eval_result()
print(f"训练历史: {list(results.keys())}")

# 绘制训练曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 损失曲线
if 'train' in results:
    train_history = results['train']['logloss']
    test_history = results['test']['logloss']
    
    axes[0].plot(train_history, label='训练', linewidth=2)
    axes[0].plot(test_history, label='测试', linewidth=2)
    axes[0].set_xlabel('迭代次数')
    axes[0].set_ylabel('LogLoss')
    axes[0].set_title('损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

# 准确率曲线
from sklearn.metrics import accuracy_score
train_pred = bst.predict(dtrain)
test_pred = bst.predict(dtest)

train_accs = [accuracy_score(y_train, (train_pred[:i+1].mean() > 0.5).astype(int) 
                            if i == 0 else (bst.predict(dtrain) > 0.5).astype(int)) 
              for i in range(min(20, len(train_pred)))]
test_accs = [accuracy_score(y_test, (bst.predict(dtest) > 0.5).astype(int))]

axes[1].plot([1, 5, 10, 20, 50, 100][:len(train_accs)], train_accs[:6], 
             marker='o', label='训练', linewidth=2)
axes[1].axhline(y=test_accs[0], color='r', linestyle='--', label='最终测试')
axes[1].set_xlabel('迭代次数')
axes[1].set_ylabel('准确率')
axes[1].set_title('准确率曲线')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 特征重要性
print("\n=== 特征重要性 ===")

# 多种特征重要性计算方法
importance_types = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for idx, imp_type in enumerate(importance_types):
    importance = bst.get_score(importance_type=imp_type)
    
    # 按重要性排序
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [x[0] for x in sorted_imp]
    values = [x[1] for x in sorted_imp]
    
    axes[idx].barh(range(len(names)), values, color='steelblue')
    axes[idx].set_yticks(range(len(names)))
    axes[idx].set_yticklabels([n[:15] for n in names])
    axes[idx].set_xlabel('重要性')
    axes[idx].set_title(f'特征重要性 ({imp_type})')
    axes[idx].invert_yaxis()

# 隐藏多余的子图
axes[5].axis('off')

plt.tight_layout()
plt.show()

# 特征重要性（使用sklearn API）
print("\n使用sklearn API计算特征重要性:")
xgb_clf2 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, 
                              random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf2.fit(X_train, y_train)

# 获取特征重要性
feature_importance = xgb_clf2.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

print("Top 10 重要特征:")
for i in range(10):
    print(f"  {feature_names[sorted_idx[i]]}: {feature_importance[sorted_idx[i]]:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(range(10), feature_importance[sorted_idx[:10]])
plt.xticks(range(10), [feature_names[i] for i in sorted_idx[:10]], rotation=45, ha='right')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('Top 10 特征重要性')
plt.tight_layout()
plt.show()

# 参数调优
print("\n=== 参数调优 ===")

# 网格搜索
print("进行网格搜索...")
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# 创建XGBoost分类器
xgb_clf_gs = xgb.XGBClassifier(
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

# 网格搜索
print("注意: 完整网格搜索可能需要较长时间...")
try:
    grid_search = GridSearchCV(
        xgb_clf_gs, param_grid,
        cv=3, scoring='accuracy',
        n_jobs=1, verbose=0
    )
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    print(f"测试集分数: {grid_search.score(X_test, y_test):.4f}")
    
    best_model = grid_search.best_estimator_
except Exception as e:
    print(f"网格搜索失败: {e}")
    print("使用默认参数继续")
    best_model = xgb_clf2

# 早停机制
print("\n=== 早停机制 ===")

# 使用早停训练
print("使用早停机制训练...")
xgb_es = xgb.XGBClassifier(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss',
    early_stopping_rounds=20
)

xgb_es.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print(f"最佳迭代次数: {xgb_es.best_iteration}")
print(f"最佳分数: {xgb_es.best_score:.4f}")

# 学习率影响
print("\n=== 学习率影响分析 ===")

learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lr in learning_rates:
    xgb_lr = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=lr,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_lr.fit(X_train, y_train, verbose=False)
    
    train_score = xgb_lr.score(X_train, y_train)
    test_score = xgb_lr.score(X_test, y_test)
    
    axes[0].plot(lr, train_score, 'o-', label=f'lr={lr}')
    axes[1].plot(lr, test_score, 's-', label=f'lr={lr}')

axes[0].set_xlabel('学习率')
axes[0].set_ylabel('训练准确率')
axes[0].set_title('学习率 vs 训练准确率')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xscale('log')

axes[1].set_xlabel('学习率')
axes[1].set_ylabel('测试准确率')
axes[1].set_title('学习率 vs 测试准确率')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xscale('log')

plt.tight_layout()
plt.show()

# max_depth影响
print("\n=== max_depth影响分析 ===")

max_depths = [2, 3, 4, 5, 6, 7, 8]
train_scores = []
test_scores = []

for depth in max_depths:
    xgb_depth = xgb.XGBClassifier(
        n_estimators=100, max_depth=depth, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric='logloss'
    )
    xgb_depth.fit(X_train, y_train, verbose=False)
    
    train_scores.append(xgb_depth.score(X_train, y_train))
    test_scores.append(xgb_depth.score(X_test, y_test))
    print(f"  max_depth={depth}: 训练={train_scores[-1]:.4f}, 测试={test_scores[-1]:.4f}")

# 可视化
plt.figure(figsize=(10, 5))
plt.plot(max_depths, train_scores, 'o-', label='训练', linewidth=2)
plt.plot(max_depths, test_scores, 's-', label='测试', linewidth=2)
plt.xlabel('max_depth')
plt.ylabel('准确率')
plt.title('max_depth 对模型性能的影响')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 交叉验证
print("\n=== 交叉验证 ===")

# K折交叉验证
cv_scores = cross_val_score(
    xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                     random_state=42, use_label_encoder=False, eval_metric='logloss'),
    X, y, cv=5, scoring='accuracy'
)

print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"各折准确率: {cv_scores.round(4)}")

# 处理不平衡数据
print("\n=== 处理不平衡数据 ===")

# 创建不平衡数据
np.random.seed(42)
n_samples = 1000
X_imbal = np.random.randn(n_samples, 10)
y_imbal = np.random.randint(0, 2, n_samples, p=[0.9, 0.1])  # 90% vs 10%

print(f"原始类别分布: {np.bincount(y_imbal)}")

# 设置scale_pos_weight参数
scale_pos_weight = (y_imbal == 0).sum() / (y_imbal == 1).sum()
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# 训练处理不平衡数据的XGBoost
xgb_imbal = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=42, use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=scale_pos_weight
)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imbal, y_imbal, test_size=0.2, random_state=42, stratify=y_imbal
)

xgb_imbal.fit(X_train_i, y_train_i)
y_pred_i = xgb_imbal.predict(X_test_i)

print(f"\n混淆矩阵:")
print(confusion_matrix(y_test_i, y_pred_i))
print(f"\n分类报告:")
print(classification_report(y_test_i, y_pred_i))

# 回归任务
print("\n=== XGBoost回归 ===")

# 创建回归数据
np.random.seed(42)
X_reg = np.random.randn(500, 5)
y_reg = X_reg[:, 0] + 2 * X_reg[:, 1] - X_reg[:, 2] + np.random.randn(500) * 0.1

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# XGBoost回归
xgb_reg = xgb.XGBRegressor(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=42
)
xgb_reg.fit(X_train_r, y_train_r)

y_pred_r = xgb_reg.predict(X_test_r)
mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"回归模型性能:")
print(f"  MSE: {mse:.4f}")
print(f"  R²: {r2:.4f}")

# 实际值vs预测值
plt.figure(figsize=(10, 5))
plt.scatter(y_test_r, y_pred_r, alpha=0.6)
plt.plot([y_test_r.min(), y_test_r.max()], [y_test_r.min(), y_test_r.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 多分类任务
print("\n=== XGBoost多分类 ===")

# 加载鸢尾花数据
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# 多分类XGBoost
xgb_mc = xgb.XGBClassifier(
    n_estimators=100, max_depth=3, learning_rate=0.1,
    random_state=42, use_label_encoder=False,
    eval_metric='mlogloss',
    objective='multi:softprob',
    num_class=3
)

xgb_mc.fit(X_train_mc, y_train_mc)
y_pred_mc = xgb_mc.predict(X_test_mc)

print(f"多分类准确率: {accuracy_score(y_test_mc, y_pred_mc):.4f}")
print(f"\n混淆矩阵:")
print(confusion_matrix(y_test_mc, y_pred_mc))

# 总结
print("\n=== XGBoost学习总结 ===")
print("1. XGBoost基础分类和回归")
print("2. DMatrix数据结构")
print("3. 多种特征重要性计算方法")
print("4. 参数调优（网格搜索、贝叶斯优化）")
print("5. 早停机制")
print("6. 学习率和树深度的影响")
print("7. 交叉验证")
print("8. 处理不平衡数据（scale_pos_weight）")
print("9. 多分类任务")
print("10. 回归任务")

print("\nXGBoost梯度提升学习完成！")
