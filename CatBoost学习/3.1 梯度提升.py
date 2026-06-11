# CatBoost梯度提升框架学习
# 主要内容：CatBoost基础、分类、回归、处理类别特征、自动调参

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from sklearn.datasets import load_breast_cancer, load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                           roc_auc_score, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')

# CatBoost基础
print("=== CatBoost基础 ===")
print(f"CatBoost版本信息: {CatBoostClassifier().get_params()}")

# 加载数据
print("\n=== 加载数据 ===")
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# CatBoost分类器
print("\n=== CatBoost分类器 ===")

# 创建CatBoost分类器
cat_clf = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False
)

# 训练模型
print("训练CatBoost分类器...")
cat_clf.fit(X_train, y_train)

# 预测
y_pred = cat_clf.predict(X_test)
y_pred_proba = cat_clf.predict_proba(X_test)[:, 1]

# 评估
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"准确率: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
print("\n=== 混淆矩阵 ===")
cm = confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns
plt.figure(figsize=(8, 6))
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
importance = cat_clf.get_feature_importance()
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': importance
}).sort_values('importance', ascending=False)

print("Top 10 重要特征:")
for i, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 可视化
plt.figure(figsize=(10, 8))
top_n = 15
top_features = feature_importance.head(top_n)
plt.barh(range(top_n), top_features['importance'].values[::-1])
plt.yticks(range(top_n), top_features['feature'].values[::-1])
plt.xlabel('重要性')
plt.ylabel('特征')
plt.title(f'CatBoost Top {top_n} 特征重要性')
plt.tight_layout()
plt.show()

# 处理类别特征
print("\n=== 处理类别特征 ===")

# 创建包含类别特征的数据集
np.random.seed(42)

# 生成数据
n_samples = 1000
data_cat = pd.DataFrame({
    'feature1': np.random.randn(n_samples),
    'feature2': np.random.randn(n_samples),
    'category1': np.random.choice(['A', 'B', 'C'], n_samples),
    'category2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
    'category3': np.random.randint(1, 6, n_samples)
})

# 创建目标变量
y_cat = (data_cat['feature1'] + data_cat['feature2'] + 
         (data_cat['category1'] == 'A').astype(int) * 0.5 +
         np.random.randn(n_samples) * 0.1 > 0).astype(int)

print("数据集信息:")
print(data_cat.head())
print(f"\n类别特征:")
print(f"  category1: {data_cat['category1'].unique()}")
print(f"  category2: {data_cat['category2'].unique()}")
print(f"  category3: {data_cat['category3'].unique()}")

# 定义类别特征的索引
cat_features = [2, 3, 4]  # category1, category2, category3的索引

# 划分数据
X_train_cat = data_cat.iloc[:800]
X_test_cat = data_cat.iloc[800:]
y_train_cat = y_cat[:800]
y_test_cat = y_cat[800:]

# 创建CatBoost分类器（使用Pool）
print("\n使用Pool处理类别特征:")
train_pool = Pool(X_train_cat, y_train_cat, cat_features=cat_features)
test_pool = Pool(X_test_cat, y_test_cat, cat_features=cat_features)

cat_clf_cat = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    random_seed=42,
    verbose=False
)

cat_clf_cat.fit(train_pool)
y_pred_cat = cat_clf_cat.predict(test_pool)
accuracy_cat = accuracy_score(y_test_cat, y_pred_cat)
print(f"带类别特征的准确率: {accuracy_cat:.4f}")

# 类别特征统计
print("\n类别特征统计:")
cat_stats = cat_clf_cat.get_object_importance()
print(f"类别特征重要性: {cat_stats}")

# 使用测试集评估
print("\n=== 使用测试集评估 ===")

# 创建评估池
eval_pool = Pool(X_test, y_test)

# 使用更多迭代和早停
cat_early = CatBoostClassifier(
    iterations=500,
    learning_rate=0.1,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=False
)

cat_early.fit(
    X_train, y_train,
    eval_set=eval_pool,
    use_best_model=True
)

print(f"最佳迭代次数: {cat_early.best_iteration_}")
print(f"最佳分数: {cat_early.best_score_}")

# 预测概率校准
print("\n=== 预测概率校准 ===")

# 查看预测概率分布
y_pred_proba_raw = cat_clf.predict_proba(X_test)[:, 1]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(y_pred_proba_raw, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.title('预测概率分布')

plt.subplot(1, 2, 2)
plt.hist(y_pred_proba, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.title('CatBoost预测概率分布')

plt.tight_layout()
plt.show()

# 回归任务
print("\n=== 回归任务 ===")

# 创建回归数据
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 创建CatBoost回归器
cat_reg = CatBoostRegressor(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    random_seed=42,
    verbose=False
)

cat_reg.fit(X_train_r, y_train_r)
y_pred_r = cat_reg.predict(X_test_r)

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

# 残差分析
residuals = y_test_r - y_pred_r
plt.figure(figsize=(10, 5))
plt.scatter(y_pred_r, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差分析')
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
cat_mc = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=False
)

cat_mc.fit(X_train_mc, y_train_mc)
y_pred_mc = cat_mc.predict(X_test_mc)

print(f"多分类准确率: {accuracy_score(y_test_mc, y_pred_mc):.4f}")
print(f"\n混淆矩阵:")
print(confusion_matrix(y_test_mc, y_pred_mc))

# 预测概率
y_pred_proba_mc = cat_mc.predict_proba(X_test_mc)
print(f"\n预测概率示例:")
for i in range(3):
    print(f"  样本{i+1}: 真实={y_test_mc[i]}, 预测={y_pred_mc[i]}, 概率={y_pred_proba_mc[i].round(3)}")

# 学习曲线
print("\n=== 学习曲线 ===")

train_scores = []
test_scores = []
iterations_range = [10, 20, 50, 100, 200, 300]

for n in iterations_range:
    cat_model = CatBoostClassifier(
        iterations=n,
        learning_rate=0.1,
        depth=6,
        random_seed=42,
        verbose=False
    )
    cat_model.fit(X_train, y_train)
    
    train_pred = cat_model.predict(X_train)
    test_pred = cat_model.predict(X_test)
    
    train_scores.append(accuracy_score(y_train, train_pred))
    test_scores.append(accuracy_score(y_test, test_pred))

plt.figure(figsize=(10, 5))
plt.plot(iterations_range, train_scores, 'o-', label='训练准确率', linewidth=2)
plt.plot(iterations_range, test_scores, 's-', label='测试准确率', linewidth=2)
plt.xlabel('迭代次数')
plt.ylabel('准确率')
plt.title('学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 交叉验证
print("\n=== 交叉验证 ===")

from catboost import cv

# 设置参数
params = {
    'iterations': 100,
    'learning_rate': 0.1,
    'depth': 6,
    'loss_function': 'Logloss',
    'eval_metric': 'Accuracy',
    'random_seed': 42,
    'verbose': False
}

# 进行交叉验证
cv_results = cv(
    pool=Pool(X, y),
    params=params,
    fold_count=5,
    shuffle=True
)

print(f"交叉验证结果:")
print(cv_results.tail())
print(f"\n最佳验证分数: {cv_results['test-Accuracy-mean'].max():.4f}")
print(f"标准差: {cv_results['test-Accuracy-std'].min():.4f}")

# 超参数网格
print("\n=== 超参数搜索 ===")

# 简化的参数搜索
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'iterations': [100, 200]
}

print("参数搜索:")
best_score = 0
best_params = {}

for depth in param_grid['depth']:
    for lr in param_grid['learning_rate']:
        for iters in param_grid['iterations']:
            cat_model = CatBoostClassifier(
                iterations=iters,
                learning_rate=lr,
                depth=depth,
                random_seed=42,
                verbose=False
            )
            
            scores = cross_val_score(cat_model, X, y, cv=3, scoring='accuracy')
            mean_score = scores.mean()
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'depth': depth, 'learning_rate': lr, 'iterations': iters}
            
            print(f"  depth={depth}, lr={lr}, iters={iters}: {mean_score:.4f}")

print(f"\n最佳参数: {best_params}")
print(f"最佳分数: {best_score:.4f}")

# 处理不平衡数据
print("\n=== 处理不平衡数据 ===")

# 创建不平衡数据
np.random.seed(42)
n_samples = 1000
X_imbal = np.random.randn(n_samples, 10)
y_imbal = np.random.randint(0, 2, n_samples, p=[0.9, 0.1])

print(f"原始类别分布: {np.bincount(y_imbal)}")

# 使用class_weights参数
cat_imbal = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    auto_class_weights='Balanced',  # 自动平衡类别权重
    random_seed=42,
    verbose=False
)

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_imbal, y_imbal, test_size=0.2, random_state=42, stratify=y_imbal
)

cat_imbal.fit(X_train_i, y_train_i)
y_pred_i = cat_imbal.predict(X_test_i)

print(f"混淆矩阵:")
print(confusion_matrix(y_test_i, y_pred_i))
print(f"分类报告:")
print(classification_report(y_test_i, y_pred_i))

# 模型保存和加载
print("\n=== 模型保存和加载 ===")

# 保存模型
cat_clf.save_model('catboost_model.cbm')
print("模型已保存为 catboost_model.cbm")

# 加载模型
loaded_model = CatBoostClassifier()
loaded_model.load_model('catboost_model.cbm')
print("模型已加载")

# 使用加载的模型预测
y_pred_loaded = loaded_model.predict(X_test)
print(f"加载模型准确率: {accuracy_score(y_test, y_pred_loaded):.4f}")

# 清理
import os
if os.path.exists('catboost_model.cbm'):
    os.remove('catboost_model.cbm')
    print("临时文件已清理")

# 性能对比
print("\n=== 性能对比 ===")

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

models = {
    'CatBoost': CatBoostClassifier(iterations=100, learning_rate=0.1, random_seed=42, verbose=False),
    'LightGBM': LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=-1),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

print("模型性能对比:")
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"  {name}: 准确率={acc:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(results.keys(), results.values(), color=['steelblue', 'coral', 'green'])
plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('模型性能对比')
plt.xticks(rotation=45, ha='right')
plt.ylim([min(results.values()) - 0.05, 1.0])
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# SHAP值
print("\n=== SHAP值解释 ===")
try:
    import shap
    
    # 创建SHAP解释器
    explainer = shap.TreeExplainer(cat_clf)
    shap_values = explainer.shap_values(X_test)
    
    # 如果是多分类，取第一个类的SHAP值
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # 可视化
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.title('SHAP特征重要性')
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("SHAP库未安装，跳过SHAP解释")

# 总结
print("\n=== CatBoost学习总结 ===")
print("1. CatBoost基础分类和回归")
print("2. 自动处理类别特征")
print("3. 特征重要性分析")
print("4. 早停机制")
print("5. 多分类任务")
print("6. 交叉验证")
print("7. 超参数搜索")
print("8. 处理不平衡数据")
print("9. 模型保存和加载")
print("10. SHAP模型解释")

print("\nCatBoost梯度提升框架学习完成！")
