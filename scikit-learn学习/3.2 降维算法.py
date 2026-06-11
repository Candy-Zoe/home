# scikit-learn降维算法学习
# 主要内容：PCA、LDA、t-SNE、UMAP降维

# 导入必要的库
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 加载数据集
print("=== 加载数据集 ===")

# 加载鸢尾花数据集
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names = iris.feature_names

print(f"鸢尾花数据形状: X={X_iris.shape}, y={y_iris.shape}")
print(f"特征名称: {feature_names}")

# 加载手写数字数据集
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"\n手写数字数据形状: X={X_digits.shape}, y={y_digits.shape}")

# 标准化数据
scaler = StandardScaler()
X_iris_scaled = scaler.fit_transform(X_iris)
X_digits_scaled = scaler.fit_transform(X_digits)

# PCA 降维
print("\n=== PCA 主成分分析 ===")

# 创建PCA模型（保留2个主成分）
pca = PCA(n_components=2)

# 拟合并转换数据
X_iris_pca = pca.fit_transform(X_iris_scaled)

print(f"PCA降维后形状: {X_iris_pca.shape}")
print(f"各主成分解释的方差比例: {pca.explained_variance_ratio_}")
print(f"累计解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")

# 查看主成分载荷
print("\n主成分载荷（各特征对主成分的贡献）:")
components_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1', 'PC2'],
    index=feature_names
)
print(components_df)

# 确定最优主成分数量
print("\n=== 确定最优主成分数量 ===")

pca_full = PCA().fit(X_iris_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-')
plt.axhline(y=0.95, color='r', linestyle='--', label='95%方差解释')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差比例')
plt.title('PCA累计解释方差')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 找到解释95%方差所需的主成分数
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"解释95%方差所需的主成分数: {n_components_95}")

# LDA 降维（监督降维）
print("\n=== LDA 线性判别分析 ===")

# 创建LDA模型（最多保留n_classes-1个成分）
lda = LDA(n_components=2)

# 拟合并转换数据
X_iris_lda = lda.fit_transform(X_iris_scaled, y_iris)

print(f"LDA降维后形状: {X_iris_lda.shape}")
print(f"各成分解释的方差比例: {lda.explained_variance_ratio_}")
print(f"累计解释方差比例: {lda.explained_variance_ratio_.sum():.4f}")

# 可视化降维结果
print("\n=== 可视化降维结果 ===")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# PCA 可视化
scatter1 = axes[0].scatter(X_iris_pca[:, 0], X_iris_pca[:, 1],
                              c=y_iris, cmap='viridis', edgecolors='black', alpha=0.7)
axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
axes[0].set_title('PCA降维结果')
plt.colorbar(scatter1, ax=axes[0], label='类别')

# LDA 可视化
scatter2 = axes[1].scatter(X_iris_lda[:, 0], X_iris_lda[:, 1],
                              c=y_iris, cmap='viridis', edgecolors='black', alpha=0.7)
axes[1].set_xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%})')
axes[1].set_ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%})')
axes[1].set_title('LDA降维结果')
plt.colorbar(scatter2, ax=axes[1], label='类别')

plt.tight_layout()
plt.show()

# t-SNE 降维
print("\n=== t-SNE 降维 ===")

# 创建t-SNE模型
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)

# 转换数据
X_digits_tsne = tsne.fit_transform(X_digits_scaled)

print(f"t-SNE降维后形状: {X_digits_tsne.shape}")

# 可视化t-SNE结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1],
                       c=y_digits, cmap='tab10', alpha=0.6)
plt.colorbar(scatter, label='数字')
plt.title('t-SNE手写数字可视化')
plt.xlabel('t-SNE 维度1')
plt.ylabel('t-SNE 维度2')
plt.show()

# 比较不同perplexity的t-SNE效果
print("\n=== 不同perplexity的t-SNE效果 ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
perplexities = [5, 15, 30, 50]

for ax, perplexity in zip(axes.flat, perplexities):
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=500)
    X_tsne = tsne.fit_transform(X_digits_scaled[:500])  # 使用子集加速

    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1],
                          c=y_digits[:500], cmap='tab10', alpha=0.6)
    ax.set_title(f't-SNE (perplexity={perplexity})')
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')

plt.tight_layout()
plt.show()

# TruncatedSVD（用于稀疏矩阵）
print("\n=== TruncatedSVD ===")

from scipy.sparse import csr_matrix

# 创建稀疏矩阵
X_sparse = csr_matrix(X_digits_scaled)

# 创建TruncatedSVD模型
svd = TruncatedSVD(n_components=2)

# 转换数据
X_digits_svd = svd.fit_transform(X_sparse)

print(f"TruncatedSVD降维后形状: {X_digits_svd.shape}")
print(f"解释方差比例: {svd.explained_variance_ratio_.sum():.4f}")

# IncrementalPCA（用于大规模数据）
print("\n=== IncrementalPCA ===")

# 分批处理数据
ipca = IncrementalPCA(n_components=2, batch_size=100)

# 分批拟合
for i in range(0, len(X_digits_scaled), 100):
    ipca.partial_fit(X_digits_scaled[i:i+100])

# 转换数据
X_digits_ipca = ipca.transform(X_digits_scaled)

print(f"IncrementalPCA降维后形状: {X_digits_ipca.shape}")
print(f"解释方差比例: {ipca.explained_variance_ratio_.sum():.4f}")

# 综合降维方法对比
print("\n=== 降维方法对比 ===")

methods = {
    'PCA': X_iris_pca,
    'LDA': X_iris_lda
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, (name, X_transformed) in zip(axes, methods.items()):
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                          c=y_iris, cmap='viridis', edgecolors='black', alpha=0.7)
    ax.set_title(f'{name}降维结果')
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')
    plt.colorbar(scatter, ax=ax, label='类别')

plt.tight_layout()
plt.show()

# 使用降维进行分类
print("\n=== 使用降维进行分类 ===")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 原始数据分类
X_train, X_test, y_train, y_test = train_test_split(
    X_iris_scaled, y_iris, test_size=0.3, random_state=42
)

clf = LogisticRegression(max_iter=200, random_state=42)
clf.fit(X_train, y_train)
accuracy_full = accuracy_score(y_test, clf.predict(X_test))
print(f"原始数据分类准确率: {accuracy_full:.4f}")

# PCA降维后分类
X_train_pca, X_test_pca = X_train @ pca.components_.T, X_test @ pca.components_.T
clf_pca = LogisticRegression(max_iter=200, random_state=42)
clf_pca.fit(X_train_pca, y_train)
accuracy_pca = accuracy_score(y_test, clf_pca.predict(X_test_pca))
print(f"PCA降维后分类准确率: {accuracy_pca:.4f}")

# LDA降维后分类
lda_clf = LDA(n_components=2)
X_train_lda = lda_clf.fit_transform(X_train, y_train)
X_test_lda = lda_clf.transform(X_test)
clf_lda = LogisticRegression(max_iter=200, random_state=42)
clf_lda.fit(X_train_lda, y_train)
accuracy_lda = accuracy_score(y_test, clf_lda.predict(X_test_lda))
print(f"LDA降维后分类准确率: {accuracy_lda:.4f}")

print("\n降维完成！")