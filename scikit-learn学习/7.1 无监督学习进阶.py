# scikit-learn无监督学习进阶学习
# 主要内容：聚类评估、降维可视化、异常检测、关联规则

from sklearn.datasets import make_blobs, make_moons, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA, KernelPCA, NMF, FastICA
from sklearn.manifold import TSNE, MDS, Isomap
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import numpy as np
import matplotlib.pyplot as plt

print("=== 生成数据 ===")
X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)
X = StandardScaler().fit_transform(X)

print(f"数据形状: {X.shape}")

print("\n=== K-Means聚类 ===")
kmeans = KMeans(n_clusters=4, random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', s=200, marker='x')
plt.title('K-Means聚类')
plt.show()

print("\n=== 聚类评估 ===")
silhouette = silhouette_score(X, y_kmeans)
calinski = calinski_harabasz_score(X, y_kmeans)
davies = davies_bouldin_score(X, y_kmeans)

print(f"轮廓系数: {silhouette:.4f}")
print(f"Calinski-Harabasz指数: {calinski:.4f}")
print(f"Davies-Bouldin指数: {davies:.4f}")

print("\n=== DBSCAN聚类 ===")
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_dbscan = dbscan.fit_predict(X_moons)

plt.figure(figsize=(10, 5))
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_dbscan, cmap='viridis')
plt.title('DBSCAN聚类')
plt.show()

print("\n=== 层次聚类 ===")
agg = AgglomerativeClustering(n_clusters=4)
y_agg = agg.fit_predict(X)

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_agg, cmap='viridis')
plt.title('层次聚类')
plt.show()

print("\n=== PCA降维 ===")
iris = load_iris()
X_iris = iris.data

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_iris)

print(f"解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差: {sum(pca.explained_variance_ratio_):.4f}")

plt.figure(figsize=(10, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=iris.target, cmap='viridis')
plt.title('PCA降维')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
plt.show()

print("\n=== t-SNE降维 ===")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_iris)

plt.figure(figsize=(10, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=iris.target, cmap='viridis')
plt.title('t-SNE降维')
plt.show()

print("\n=== Kernel PCA ===")
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_moons)

plt.figure(figsize=(10, 5))
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_dbscan, cmap='viridis')
plt.title('Kernel PCA降维')
plt.show()

print("\n=== 异常检测 ===")
X_outliers = np.random.randn(300, 2)
X_outliers[-10:] = np.random.uniform(low=-4, high=4, size=(10, 2))

iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_iso = iso_forest.fit_predict(X_outliers)

plt.figure(figsize=(10, 5))
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_iso, cmap='coolwarm')
plt.title('Isolation Forest异常检测')
plt.show()

print("\n=== Local Outlier Factor ===")
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
y_lof = lof.fit_predict(X_outliers)

plt.figure(figsize=(10, 5))
plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c=y_lof, cmap='coolwarm')
plt.title('LOF异常检测')
plt.show()

print("\n=== 肘部法则 ===")
inertias = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(K, inertias, 'bx-')
plt.xlabel('聚类数 k')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.show()