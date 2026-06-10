# scikit-learn聚类算法学习
# 主要内容：K-Means、层次聚类、DBSCAN

from sklearn.datasets import make_blobs, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
import matplotlib.pyplot as plt

print("=== 创建聚类数据 ===")
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=42)

plt.scatter(X[:, 0], X[:, 1])
plt.title('原始数据')
plt.show()

print("\n=== K-Means聚类 ===")
kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', color='red')
plt.title('K-Means聚类结果')
plt.show()

print("\n=== 层次聚类 ===")
agg = AgglomerativeClustering(n_clusters=4)
labels_agg = agg.fit_predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels_agg, cmap='viridis')
plt.title('层次聚类结果')
plt.show()

print("\n=== DBSCAN聚类 ===")
X_moons, _ = make_moons(n_samples=200, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_db = dbscan.fit_predict(X_moons)

plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_db, cmap='viridis')
plt.title('DBSCAN聚类结果')
plt.show()

print("\n=== 肘部法则选择K值 ===")
inertias = []
k_range = range(1, 10)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(k_range, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('惯性')
plt.title('肘部法则')
plt.show()