# scikit-learn特征工程学习
# 主要内容：特征提取、特征选择、特征转换、特征交互

from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np

print("=== 加载数据集 ===")
iris = load_iris()
X, y = iris.data, iris.target
print(f"原始特征数: {X.shape[1]}")

print("\n=== 单变量特征选择 ===")
selector = SelectKBest(f_classif, k=2)
X_new = selector.fit_transform(X, y)
print(f"选择后特征数: {X_new.shape[1]}")
print(f"特征得分: {selector.scores_}")
print(f"被选中的特征索引: {selector.get_support(indices=True)}")

print("\n=== 递归特征消除 ===")
estimator = RandomForestClassifier(n_estimators=10)
rfe = RFE(estimator, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
print(f"RFE选择后特征数: {X_rfe.shape[1]}")
print(f"特征排名: {rfe.ranking_}")

print("\n=== PCA降维 ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {np.sum(pca.explained_variance_ratio_):.4f}")

print("\n=== NMF非负矩阵分解 ===")
nmf = NMF(n_components=2, init='random', random_state=0)
X_nmf = nmf.fit_transform(X)
print(f"NMF重构误差: {nmf.reconstruction_err_:.4f}")

print("\n=== 多项式特征 ===")
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
print(f"原始特征数: {X.shape[1]}")
print(f"多项式特征数: {X_poly.shape[1]}")
print(f"特征名称: {poly.get_feature_names_out()}")

print("\n=== 分箱处理 ===")
kbins = KBinsDiscretizer(n_bins=3, encode='onehot-dense', strategy='uniform')
X_binned = kbins.fit_transform(X[:, :1])
print(f"分箱后特征数: {X_binned.shape[1]}")

print("\n=== 文本特征提取 ===")
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?'
]

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(corpus)
print(f"词汇表大小: {len(vectorizer.vocabulary_)}")
print(f"TF-IDF矩阵形状: {X_tfidf.shape}")
print(f"特征名称: {vectorizer.get_feature_names_out()[:5]}")

print("\n=== 管道组合 ===")
pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('select', SelectKBest(f_classif, k=5)),
    ('classifier', RandomForestClassifier())
])

pipeline.fit(X, y)
accuracy = pipeline.score(X, y)
print(f"管道准确率: {accuracy:.4f}")