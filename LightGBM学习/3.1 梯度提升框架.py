# LightGBM梯度提升框架学习
# 主要内容：LightGBM基础、分类、回归、参数调优、特征重要性、API详解

# 导入必要的库
import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_auc_score, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# LightGBM基础
print("=== LightGBM基础 ===")
print(f"LightGBM版本: {lgb.__version__}")

# 加载数据
print("\n=== 加载数据 ===")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

# LightGBM分类器
print("\n=== LightGBM分类器 ===")

# 创建LightGBM分类器
lgb_clf = lgb.LGBMClassifier(
    objective='binary',
    metric='binary_logloss',
    boosting_type='gbdt',
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=100,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1
)

# 训练模型
print("训练LightGBM分类器...")
lgb_clf.fit(X_train, y_train)

# 预测
y_pred = lgb_clf.predict(X_test)
y_pred_proba = lgb_clf.predict_proba(X_test)[:, 1]

# 评估
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"准确率: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵可视化
print("\n=== 混淆矩阵 ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
import seaborn as sns
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['负类', '正类'], yticklabels=['负类', '正类'])
plt.xlabel('预测')
plt.ylabel('实际')
plt.title('混淆矩阵')
plt.tight_layout()
plt.show()

# 特征重要性
print("\n=== 特征重要性 ===")

# 获取特征重要性
importance = lgb_clf.feature_importances_
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Top 10 重要特征:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 8))
top_n = 20
top_features = feature_importance.head(top_n)
plt.barh(range(top_n), top_features['importance'].values[::-1])
plt.yticks(range(top_n), top_features['feature'].values[::-1])
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title(f'LightGBM Top {top_n} 特征重要性')
plt.tight_layout()
plt.show()

# 多种boosting类型
print("\n=== 多种Boosting类型 ===")

boosting_types = ['gbdt', 'dart', 'rf']  # GBDT、DART、随机森林

results = {}
for boosting_type in boosting_types:
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type=boosting_type,
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    )
    
    lgb_model.fit(X_train, y_train)
    y_pred = lgb_model.predict(X_test)
    y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_proba)
    
    results[boosting_type] = {'accuracy': acc, 'roc_auc': roc}
    print(f"{boosting_type.upper()}: 准确率={acc:.4f}, ROC AUC={roc:.4f}")

# 可视化不同boosting类型的效果
plt.figure(figsize=(10, 5))
x = np.arange(len(boosting_types))
width = 0.35

accs = [results[b]['accuracy'] for b in boosting_types]
rocs = [results[b]['roc_auc'] for b in boosting_types]

plt.bar(x - width/2, accs, width, label='准确率')
plt.bar(x + width/2, rocs, width, label='ROC AUC')
plt.xlabel('Boosting类型')
plt.ylabel('分数')
plt.title('不同Boosting类型对比')
plt.xticks(x, [b.upper() for b in boosting_types])
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 参数调优
print("\n=== 参数调优 ===")

# 创建参数字典
param_grid = {
    'num_leaves': [15, 31, 63],
    'max_depth': [3, 5, 7, -1],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200]
}

# 简化网格搜索
print("进行简化网格搜索（演示用）...")
best_score = 0
best_params = {}

for num_leaves in [15, 31, 63]:
    for max_depth in [3, 5, -1]:
        for learning_rate in [0.05, 0.1]:
            lgb_model = lgb.LGBMClassifier(
                objective='binary',
                num_leaves=num_leaves,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_estimators=100,
                random_state=42,
                verbosity=-1
            )
            
            scores = cross_val_score(lgb_model, X_train, y_train, cv=3, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    'num_leaves': num_leaves,
                    'max_depth': max_depth,
                    'learning_rate': learning_rate
                }

print(f"最佳参数: {best_params}")
print(f"最佳交叉验证分数: {best_score:.4f}")

# 使用最佳参数训练
lgb_best = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=100,
    random_state=42,
    verbosity=-1,
    **best_params
)
lgb_best.fit(X_train, y_train)
y_pred_best = lgb_best.predict(X_test)
print(f"测试集准确率: {accuracy_score(y_test, y_pred_best):.4f}")

# 学习曲线
print("\n=== 学习曲线 ===")

train_scores = []
test_scores = []
n_estimators_range = [10, 20, 50, 100, 200, 300]

for n in n_estimators_range:
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=n,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        verbosity=-1
    )
    lgb_model.fit(X_train, y_train)
    
    train_pred = lgb_model.predict(X_train)
    test_pred = lgb_model.predict(X_test)
    
    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

plt.figure(figsize=(10, 5))
plt.plot(n_estimators_range, train_scores, 'o-', label='训练准确率', linewidth=2)
plt.plot(n_estimators_range, test_scores, 's-', label='测试准确率', linewidth=2)
plt.xlabel('n_estimators')
plt.ylabel('准确率')
plt.title('学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 早停机制
print("\n=== 早停机制 ===")

# 使用早停训练
lgb_early = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=1000,  # 设置较大的值
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

# 使用回调进行早停
lgb_early.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=50, verbose=True),
        lgb.log_evaluation(period=100)
    ]
)

print(f"最佳迭代次数: {lgb_early.best_iteration_}")
print(f"最佳分数: {lgb_early.best_score_}")

# 回归任务
print("\n=== 回归任务 ===")

# 创建回归数据
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 创建LightGBM回归器
lgb_reg = lgb.LGBMRegressor(
    objective='regression',
    metric='rmse',
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

lgb_reg.fit(X_train_r, y_train_r)
y_pred_r = lgb_reg.predict(X_test_r)

mse = mean_squared_error(y_test_r, y_pred_r)
r2 = r2_score(y_test_r, y_pred_r)

print(f"回归模型性能:")
print(f"  MSE: {mse:.4f}")
print(f"  RMSE: {np.sqrt(mse):.4f}")
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
print("\n=== 多分类任务 ===")

# 加载鸢尾花数据
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_train_mc, X_test_mc, y_train_mc, y_test_mc = train_test_split(
    X_iris, y_iris, test_size=0.2, random_state=42, stratify=y_iris
)

# 创建多分类器
lgb_mc = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=100,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    verbosity=-1
)

lgb_mc.fit(X_train_mc, y_train_mc)
y_pred_mc = lgb_mc.predict(X_test_mc)

print(f"多分类准确率: {accuracy_score(y_test_mc, y_pred_mc):.4f}")
print(f"\n混淆矩阵:")
print(confusion_matrix(y_test_mc, y_pred_mc))

# 特征重要性（多分类）
print("\n=== 多分类特征重要性 ===")

importance_mc = lgb_mc.feature_importances_
feature_importance_mc = pd.DataFrame({
    'feature': iris.feature_names,
    'importance': importance_mc
}).sort_values('importance', ascending=False)

print("特征重要性排序:")
for _, row in feature_importance_mc.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(range(len(importance_mc)), importance_mc)
plt.xticks(range(len(importance_mc)), iris.feature_names, rotation=45, ha='right')
plt.xlabel('特征')
plt.ylabel('重要性')
plt.title('多分类特征重要性')
plt.tight_layout()
plt.show()

# 交叉验证
print("\n=== 交叉验证 ===")

# 5折交叉验证
cv_scores = cross_val_score(
    lgb.LGBMClassifier(
        objective='binary',
        n_estimators=100,
        learning_rate=0.05,
        random_state=42,
        verbosity=-1
    ),
    X, y, cv=5, scoring='accuracy'
)

print(f"5折交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
print(f"各折准确率: {cv_scores.round(4)}")

# 分层K折交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_skf = cross_val_score(
    lgb.LGBMClassifier(objective='binary', n_estimators=100, random_state=42, verbosity=-1),
    X, y, cv=skf, scoring='roc_auc'
)

print(f"\n分层5折交叉验证ROC AUC: {cv_scores_skf.mean():.4f} (+/- {cv_scores_skf.std():.4f})")

# 自定义目标函数
print("\n=== 自定义目标函数 ===")

# 自定义对数损失
def custom_binary_objective(y_true, y_pred):
    """自定义二分类目标函数"""
    # 计算梯度
    grad = y_pred - y_true
    # 计算二阶导数
    hess = y_pred * (1 - y_pred)
    return grad, hess

# 自定义评估指标
def custom_eval_metric(y_true, y_pred):
    """自定义评估指标"""
    pred_class = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, pred_class)
    return 'custom_acc', acc, True

# 使用自定义目标训练
lgb_custom = lgb.LGBMClassifier(
    objective=custom_binary_objective,
    n_estimators=100,
    learning_rate=0.05,
    random_state=42,
    verbosity=-1
)

lgb_custom.fit(X_train, y_train, eval_metric=custom_eval_metric)
y_pred_custom = lgb_custom.predict(X_test)
print(f"自定义目标准确率: {accuracy_score(y_test, y_pred_custom):.4f}")

# 处理不平衡数据
print("\n=== 处理不平衡数据 ===")

# 创建不平衡数据
np.random.seed(42)
n_samples = 1000
X_imbal = np.random.randn(n_samples, 10)
y_imbal = np.random.randint(0, 2, n_samples, p=[0.95, 0.05])

print(f"原始类别分布: {np.bincount(y_imbal)}")

# 使用class_weight参数
lgb_imbal = lgb.LGBMClassifier(
    objective='binary',
    n_estimators=100,
    learning_rate=0.05,
    class_weight='balanced',  # 自动平衡类别权重
    random_state=42,
    verbosity=-1
)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imbal, y_imbal, test_size=0.2, random_state=42, stratify=y_imbal
)

lgb_imbal.fit(X_train_i, y_train_i)
y_pred_i = lgb_imbal.predict(X_test_i)

print(f"混淆矩阵:")
print(confusion_matrix(y_test_i, y_pred_i))
print(f"分类报告:")
print(classification_report(y_test_i, y_pred_i))

# SHAP值
print("\n=== SHAP值解释 ===")
try:
    import shap
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(lgb_clf)
    shap_values = explainer.shap_values(X_test)
    
    # 可视化SHAP值
    shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names, show=False)
    plt.title('SHAP特征重要性')
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("SHAP库未安装，跳过SHAP解释")

# 性能对比
print("\n=== 性能对比 ===")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# 数据标准化（用于逻辑回归）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

print("模型性能对比:")
results_compare = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results_compare[name] = acc
    print(f"  {name}: 准确率={acc:.4f}")

# 可视化对比
plt.figure(figsize=(10, 5))
plt.bar(results_compare.keys(), results_compare.values(), color=['steelblue', 'coral', 'green', 'purple'])
plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('模型性能对比')
plt.xticks(rotation=45, ha='right')
plt.ylim([min(results_compare.values()) - 0.05, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# 总结
print("\n=== LightGBM学习总结 ===")
print("1. LightGBM基础分类和回归")
print("2. 多种boosting类型（GBDT、DART、RF）")
print("3. 参数调优和网格搜索")
print("4. 早停机制")
print("5. 特征重要性分析")
print("6. 多分类任务")
print("7. 交叉验证")
print("8. 自定义目标函数和评估指标")
print("9. 处理不平衡数据")
print("10. SHAP模型解释")
print("11. 性能对比")

print("\nLightGBM梯度提升框架学习完成！")
