# scikit-learn高级特征工程学习
# 主要内容：特征选择、特征变换、特征构造、特征编码、特征缩放

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder,
    PolynomialFeatures, KBinsDiscretizer,
    PowerTransformer, QuantileTransformer,
    FunctionTransformer
)
from sklearn.feature_selection import (
    SelectKBest, SelectPercentile, SelectFromModel,
    chi2, f_classif, f_regression,
    VarianceThreshold, RFE, RFECV,
    mutual_info_classif, mutual_info_regression
)
from sklearn.decomposition import PCA, NMF, TruncatedSVD
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error

# 加载数据集
print("=== 加载数据集 ===")

# 分类数据集
cancer = load_breast_cancer()
X_cancer = cancer.data
y_cancer = cancer.target
feature_names_cancer = cancer.feature_names

# 回归数据集（使用替代数据）
np.random.seed(42)
X_regression = np.random.randn(500, 10)
y_regression = np.random.randn(500)

print(f"分类数据集形状: X={X_cancer.shape}, y={y_cancer.shape}")
print(f"回归数据集形状: X={X_regression.shape}, y={y_regression.shape}")

# 划分数据集
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cancer, y_cancer, test_size=0.2, random_state=42
)

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_regression, y_regression, test_size=0.2, random_state=42
)

print(f"分类训练集大小: {X_train_c.shape}")
print(f"分类测试集大小: {X_test_c.shape}")

# 特征缩放
print("\n=== 特征缩放 ===")

# 1. StandardScaler (Z-score标准化)
print("1. StandardScaler (Z-score标准化)")
scaler_standard = StandardScaler()
X_train_standard = scaler_standard.fit_transform(X_train_c)
X_test_standard = scaler_standard.transform(X_test_c)

print(f"训练集均值: {X_train_standard.mean(axis=0)[:3].round(4)}")
print(f"训练集标准差: {X_train_standard.std(axis=0)[:3].round(4)}")

# 2. MinMaxScaler (最小-最大缩放)
print("\n2. MinMaxScaler (最小-最大缩放)")
scaler_minmax = MinMaxScaler()
X_train_minmax = scaler_minmax.fit_transform(X_train_c)
X_test_minmax = scaler_minmax.transform(X_test_c)

print(f"训练集最小值: {X_train_minmax.min(axis=0)[:3].round(4)}")
print(f"训练集最大值: {X_train_minmax.max(axis=0)[:3].round(4)}")

# 3. RobustScaler (鲁棒缩放)
print("\n3. RobustScaler (鲁棒缩放)")
scaler_robust = RobustScaler()
X_train_robust = scaler_robust.fit_transform(X_train_c)
X_test_robust = scaler_robust.transform(X_test_c)

print("鲁棒缩放完成（对异常值不敏感）")

# 4. MaxAbsScaler (最大绝对值缩放)
from sklearn.preprocessing import MaxAbsScaler
print("\n4. MaxAbsScaler (最大绝对值缩放)")
scaler_maxabs = MaxAbsScaler()
X_train_maxabs = scaler_maxabs.fit_transform(X_train_c)
X_test_maxabs = scaler_maxabs.transform(X_test_c)

print(f"训练集最大绝对值: {X_train_maxabs.max(axis=0)[:3].round(4)}")

# 可视化不同缩放方法
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes[0, 0].scatter(X_train_c[:, 0], X_train_c[:, 1], c=y_train_c, cmap='viridis', alpha=0.6)
axes[0, 0].set_title('原始数据')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(X_train_standard[:, 0], X_train_standard[:, 1], c=y_train_c, cmap='viridis', alpha=0.6)
axes[0, 1].set_title('StandardScaler')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(X_train_minmax[:, 0], X_train_minmax[:, 1], c=y_train_c, cmap='viridis', alpha=0.6)
axes[1, 0].set_title('MinMaxScaler')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(X_train_robust[:, 0], X_train_robust[:, 1], c=y_train_c, cmap='viridis', alpha=0.6)
axes[1, 1].set_title('RobustScaler')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('不同特征缩放方法对比')
plt.tight_layout()
plt.show()

# 类别特征编码
print("\n=== 类别特征编码 ===")

# 创建示例数据
data = {
    '颜色': ['红色', '蓝色', '绿色', '红色', '蓝色', '绿色'],
    '尺寸': ['小', '中', '大', '小', '中', '大'],
    '价格': [100, 200, 300, 150, 250, 350],
    '类别': ['A', 'B', 'A', 'B', 'A', 'B']
}
df = pd.DataFrame(data)

print("示例数据:")
print(df)

# 1. One-Hot Encoding (独热编码)
print("\n1. One-Hot Encoding")
encoder_onehot = OneHotEncoder(sparse=False)
onehot_encoded = encoder_onehot.fit_transform(df[['颜色', '尺寸']])
print(f"独热编码后形状: {onehot_encoded.shape}")
print(f"编码后的特征: {encoder_onehot.get_feature_names_out(['颜色', '尺寸'])}")

# 2. Label Encoding (标签编码)
print("\n2. Label Encoding")
encoder_label = LabelEncoder()
label_encoded = encoder_label.fit_transform(df['颜色'])
print(f"标签编码: {label_encoded}")
print(f"类别映射: {dict(zip(encoder_label.classes_, range(len(encoder_label.classes_))))}")

# 3. Ordinal Encoding (顺序编码)
print("\n3. Ordinal Encoding")
encoder_ordinal = OrdinalEncoder(categories=[['小', '中', '大']])
ordinal_encoded = encoder_ordinal.fit_transform(df[['尺寸']])
print(f"顺序编码: {ordinal_encoded.flatten()}")
print(f"类别映射: 小=0, 中=1, 大=2")

# 4. Target Encoding (目标编码)
print("\n4. Target Encoding (手动实现)")
# 使用类别特征的目标值均值进行编码
df_encoded = df.copy()
target_map = df.groupby('颜色')['类别'].map(lambda x: (x == 'A').mean())
df_encoded['颜色_TE'] = df['颜色'].map(target_map)
print(f"目标编码结果:\n{df_encoded[['颜色', '颜色_TE']]}")

# 5. Frequency Encoding (频率编码)
print("\n5. Frequency Encoding")
freq_map = df['颜色'].value_counts(normalize=True).to_dict()
df_encoded['颜色_FE'] = df['颜色'].map(freq_map)
print(f"频率编码结果:\n{df_encoded[['颜色', '颜色_FE']]}")

# 特征选择
print("\n=== 特征选择 ===")

# 方法1: 方差阈值
print("1. 方差阈值")
selector_variance = VarianceThreshold(threshold=0.1)
X_train_variance = selector_variance.fit_transform(X_train_c)
print(f"原始特征数: {X_train_c.shape[1]}")
print(f"选择后特征数: {X_train_variance.shape[1]}")

# 方法2: 单变量特征选择
print("\n2. 单变量特征选择 (SelectKBest)")
selector_kbest = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = selector_kbest.fit_transform(X_train_c, y_train_c)
print(f"选择的特征索引: {selector_kbest.get_support(indices=True)}")
print(f"前10个特征的F值: {selector_kbest.scores_[selector_kbest.get_support(indices=True)][:5].round(2)}")

# 方法3: 百分位数选择
print("\n3. 百分位数选择 (SelectPercentile)")
selector_percentile = SelectPercentile(score_func=f_classif, percentile=30)
X_train_percentile = selector_percentile.fit_transform(X_train_c, y_train_c)
print(f"选择后特征数: {X_train_percentile.shape[1]}")

# 方法4: 基于模型的选择
print("\n4. 基于模型的特征选择 (SelectFromModel)")
selector_model = SelectFromModel(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42),
    threshold='median'
)
X_train_model = selector_model.fit_transform(X_train_c, y_train_c)
print(f"选择的特征数: {X_train_model.shape[1]}")
print(f"重要特征: {feature_names_cancer[selector_model.get_support()][:5]}")

# 方法5: 递归特征消除
print("\n5. 递归特征消除 (RFE)")
selector_rfe = RFE(
    estimator=LogisticRegression(max_iter=1000),
    n_features_to_select=10
)
X_train_rfe = selector_rfe.fit_transform(X_train_c, y_train_c)
print(f"选择的特征数: {X_train_rfe.shape[1]}")
print(f"特征排名: {selector_rfe.ranking_[:10]}")

# 方法6: 带交叉验证的RFE
print("\n6. 带交叉验证的RFE (RFECV)")
selector_rfecv = RFECV(
    estimator=LogisticRegression(max_iter=1000),
    step=1,
    cv=5,
    min_features_to_select=5
)
X_train_rfecv = selector_rfecv.fit_transform(X_train_c, y_train_c)
print(f"最优特征数: {selector_rfecv.n_features_}")

# 特征重要性可视化
print("\n=== 特征重要性可视化 ===")

# 使用随机森林获取特征重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_c, y_train_c)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

# 绘制特征重要性
plt.figure(figsize=(10, 6))
plt.title('特征重要性 (Random Forest)')
plt.bar(range(10), importances[indices[:10]], align='center')
plt.xticks(range(10), [feature_names_cancer[i] for i in indices[:10]], rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 特征构造
print("\n=== 特征构造 ===")

# 方法1: 多项式特征
print("1. 多项式特征")
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train_c[:, :5])  # 使用前5个特征
print(f"原始特征数: 5")
print(f"多项式特征数: {X_train_poly.shape[1]}")
print(f"特征名称示例: {poly.get_feature_names_out()[:5]}")

# 方法2: 分箱特征
print("\n2. 分箱特征 (KBinsDiscretizer)")
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
X_train_binned = discretizer.fit_transform(X_train_c[:, :3])
print(f"原始特征数: 3")
print(f"分箱后特征数: {X_train_binned.shape[1]}")
print(f"分箱边界:\n{discretizer.bin_edges_[0][:3]}")

# 方法3: 数学变换特征
print("\n3. 数学变换特征")
transformer_log = FunctionTransformer(np.log1p, validate=True)
transformer_sqrt = FunctionTransformer(np.sqrt, validate=True)

X_train_log = transformer_log.fit_transform(np.abs(X_train_c[:, :3]))
X_train_sqrt = transformer_sqrt.fit_transform(np.abs(X_train_c[:, :3]))

print("对数变换和平方根变换完成")

# 方法4: 统计特征
print("\n4. 统计特征")
# 为每个样本创建统计特征
X_train_stats = np.column_stack([
    X_train_c.mean(axis=1),
    X_train_c.std(axis=1),
    X_train_c.min(axis=1),
    X_train_c.max(axis=1),
    np.median(X_train_c, axis=1)
])
print(f"统计特征形状: {X_train_stats.shape}")
print("统计特征: mean, std, min, max, median")

# 特征变换
print("\n=== 特征变换 ===")

# 方法1: PCA (主成分分析)
print("1. PCA 降维")
pca = PCA(n_components=10)
X_train_pca = pca.fit_transform(X_train_standard)
print(f"原始特征数: {X_train_standard.shape[1]}")
print(f"降维后特征数: {X_train_pca.shape[1]}")
print(f"解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")

# 可视化PCA结果
plt.figure(figsize=(8, 6))
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_c, cmap='viridis', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 可视化')
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 方法2: 幂变换 (Power Transformer)
print("\n2. 幂变换 (Power Transformer)")
transformer_power = PowerTransformer(method='yeo-johnson')
X_train_power = transformer_power.fit_transform(X_train_c[:, :5])
print("幂变换完成（使数据更接近正态分布）")

# 方法3: 分位数变换
print("\n3. 分位数变换")
transformer_quantile = QuantileTransformer(n_quantiles=100, output_distribution='normal')
X_train_quantile = transformer_quantile.fit_transform(X_train_c[:, :5])
print("分位数变换完成")

# 特征选择对比
print("\n=== 不同特征选择方法对比 ===")

# 定义不同的特征选择方法
feature_selectors = {
    'All Features': None,
    'VarianceThreshold': VarianceThreshold(threshold=0.1),
    'SelectKBest (k=10)': SelectKBest(score_func=f_classif, k=10),
    'SelectKBest (k=15)': SelectKBest(score_func=f_classif, k=15),
    'SelectFromModel': SelectFromModel(RandomForestClassifier(random_state=42)),
    'PCA (10)': Pipeline([('scaler', StandardScaler()), ('pca', PCA(n_components=10))])
}

# 评估每种方法
results = {}
for name, selector in feature_selectors.items():
    if selector is None:
        X_train_selected = X_train_c
        X_test_selected = X_test_c
    else:
        X_train_selected = selector.fit_transform(X_train_c, y_train_c)
        X_test_selected = selector.transform(X_test_c)
    
    # 训练和评估
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train_selected, y_train_c)
    y_pred = clf.predict(X_test_selected)
    acc = accuracy_score(y_test_c, y_pred)
    
    results[name] = {
        'n_features': X_train_selected.shape[1],
        'accuracy': acc
    }
    print(f"{name}: 特征数={X_train_selected.shape[1]}, 准确率={acc:.4f}")

# 可视化结果
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
names = list(results.keys())
n_features = [r['n_features'] for r in results.values()]
accuracies = [r['accuracy'] for r in results.values()]

axes[0].bar(names, n_features, color='skyblue')
axes[0].set_xlabel('方法')
axes[0].set_ylabel('特征数')
axes[0].set_title('不同方法选择的特征数')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(names, accuracies, color='salmon')
axes[1].set_xlabel('方法')
axes[1].set_ylabel('准确率')
axes[1].set_title('不同方法的分类准确率')
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim([min(accuracies) - 0.05, 1.0])

plt.tight_layout()
plt.show()

# 特征工程Pipeline
print("\n=== 特征工程Pipeline ===")

# 创建完整的特征工程Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(k=15)),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 训练和评估
pipeline.fit(X_train_c, y_train_c)
y_pred = pipeline.predict(X_test_c)
accuracy = accuracy_score(y_test_c, y_pred)

print(f"Pipeline准确率: {accuracy:.4f}")

# 使用GridSearchCV优化Pipeline
from sklearn.model_selection import GridSearchCV

param_grid = {
    'feature_selection__k': [10, 15, 20],
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2']
}

print("\n网格搜索优化Pipeline...")
try:
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_c, y_train_c)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")
    print(f"测试集分数: {grid_search.score(X_test_c, y_test_c):.4f}")
except Exception as e:
    print(f"网格搜索失败: {e}")

# 特征工程最佳实践
print("\n=== 特征工程最佳实践 ===")
print("1. 数据探索和理解")
print("2. 处理缺失值和异常值")
print("3. 特征编码（类别特征、数值特征）")
print("4. 特征缩放和标准化")
print("5. 特征构造（交互项、多项式、统计特征）")
print("6. 特征选择（过滤、包装、嵌入方法）")
print("7. 降维（PCA、LDA等）")
print("8. 验证特征对模型性能的影响")
print("9. 使用Pipeline进行组合")
print("10. 定期重新评估特征重要性")

# 总结
print("\n=== scikit-learn高级特征工程学习总结 ===")
print("1. 多种特征缩放方法")
print("2. 类别特征编码技术")
print("3. 多种特征选择方法")
print("4. 特征构造方法")
print("5. 特征变换技术")
print("6. 使用Pipeline进行组合")
print("7. 特征工程最佳实践")

print("\nscikit-learn高级特征工程学习完成！")
