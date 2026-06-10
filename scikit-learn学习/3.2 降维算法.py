# scikit-learn降维算法学习
# 主要内容：PCA、t-SNE、特征选择

from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_regression
import matplotlib.pyplot as plt

print("=== 加载数据 ===")
digits = load_digits()
X = digits.data
y = digits.target
print(f"原始数据形状: {X.shape}")

print("\n=== PCA降维 ===")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"PCA后形状: {X_pca.shape}")
print(f"解释方差比: {pca.explained_variance_ratio_}")

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title('PCA降维可视化')
plt.show()

print("\n=== 累积方差 ===")
pca_full = PCA()
pca_full.fit(X)
cumulative_variance = pca_full.explained_variance_ratio_.cumsum()

plt.plot(cumulative_variance)
plt.xlabel('主成分数量')
plt.ylabel('累积解释方差')
plt.title('PCA累积方差')
plt.show()

print("\n=== t-SNE降维 ===")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')
plt.colorbar()
plt.title('t-SNE降维可视化')
plt.show()

print("\n=== 特征选择 ===")
from sklearn.datasets import make_regression
X_reg, y_reg = make_regression(n_samples=100, n_features=20, random_state=42)

selector = SelectKBest(f_regression, k=5)
X_selected = selector.fit_transform(X_reg, y_reg)
print(f"特征选择后形状: {X_selected.shape}")
print(f"选择的特征索引: {selector.get_support(indices=True)}")