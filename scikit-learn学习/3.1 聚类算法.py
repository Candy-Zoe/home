# scikit-learn聚类算法学习
# 主要内容：K-Means、DBSCAN、层次聚类、聚类评估

# 导入必要的库
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import numpy as np
import matplotlib.pyplot as plt

# 生成聚类数据
print("=== 生成聚类数据 ===")

# 生成3个簇的 blobs 数据
X_blobs, y_blobs = make_blobs(
    n_samples=300,
    centers=3,
    cluster_std=1.0,
    random_state=42
)

print(f"Blobs数据形状: X={X_blobs.shape}, y={y_blobs.shape}")
print(f"真实簇标签分布: {np.bincount(y_blobs)}")

# 生成 moon 数据（适合DBSCAN）
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
print(f"\nMoon数据形状: X={X_moons.shape}")

# 标准化数据
scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)

# K-Means 聚类
print("\n=== K-Means 聚类 ===")

# 创建K-Means模型
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)

# 训练模型
kmeans.fit(X_blobs_scaled)

# 获取聚类标签
labels_kmeans = kmeans.predict(X_blobs_scaled)

# 获取聚类中心
centers = kmeans.cluster_centers_

print(f"聚类标签: {np.unique(labels_kmeans)}")
print(f"聚类中心:\n{centers}")
print(f"惯性（簇内平方和）: {kmeans.inertia_:.4f}")

# 确定最优簇数（肘部法则）
print("\n=== 肘部法则 ===")

inertias = []
silhouettes = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_blobs_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_blobs_scaled, kmeans.labels_))

# 绘制肘部法则和轮廓系数
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_xlabel('簇数 (K)')
axes[0].set_ylabel('惯性')
axes[0].set_title('肘部法则')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouettes, 'ro-')
axes[1].set_xlabel('簇数 (K)')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数法')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# DBSCAN 聚类
print("\n=== DBSCAN 聚类 ===")

# 创建DBSCAN模型
dbscan = DBSCAN(eps=0.3, min_samples=5)

# 训练模型
labels_dbscan = dbscan.fit_predict(X_moons)

print(f"聚类标签: {np.unique(labels_dbscan)}")
print(f"噪声点数量: {np.sum(labels_dbscan == -1)}")

# 层次聚类
print("\n=== 层次聚类 ===")

# 创建层次聚类模型
agg_clustering = AgglomerativeClustering(n_clusters=3)

# 训练模型
labels_agg = agg_clustering.fit_predict(X_blobs_scaled)

print(f"聚类标签: {np.unique(labels_agg)}")

# 聚类评估
print("\n=== 聚类评估 ===")

# 轮廓系数（-1到1之间，越接近1越好）
silhouette = silhouette_score(X_blobs_scaled, labels_kmeans)
print(f"轮廓系数: {silhouette:.4f}")

# Calinski-Harabasz指数（越大越好）
calinski = calinski_harabasz_score(X_blobs_scaled, labels_kmeans)
print(f"Calinski-Harabasz指数: {calinski:.4f}")

# Davies-Bouldin指数（越小越好）
davies = davies_bouldin_score(X_blobs_scaled, labels_kmeans)
print(f"Davies-Bouldin指数: {davies:.4f}")

# 可视化聚类结果
print("\n=== 可视化聚类结果 ===")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原始数据
axes[0, 0].scatter(X_blobs[:, 0], X_blobs[:, 1], c=y_blobs, cmap='viridis')
axes[0, 0].set_title('原始数据（真实标签）')
axes[0, 0].grid(True, alpha=0.3)

# K-Means 结果
axes[0, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_kmeans, cmap='viridis')
axes[0, 1].scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='聚类中心')
axes[0, 1].set_title('K-Means 聚类结果')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# DBSCAN 结果（用于moon数据）
axes[1, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_dbscan, cmap='viridis')
axes[1, 0].set_title('DBSCAN 聚类结果（Moon数据）')
axes[1, 0].grid(True, alpha=0.3)

# 层次聚类结果
axes[1, 1].scatter(X_blobs[:, 0], X_blobs[:, 1], c=labels_agg, cmap='viridis')
axes[1, 1].set_title('层次聚类结果')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 图像分割示例
print("\n=== 聚类数量选择分析 ===")

# 对不同簇数进行评估
evaluations = []

for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_blobs_scaled)

    silhouette = silhouette_score(X_blobs_scaled, labels)
    calinski = calinski_harabasz_score(X_blobs_scaled, labels)
    davies = davies_bouldin_score(X_blobs_scaled, labels)

    evaluations.append({
        'k': k,
        'silhouette': silhouette,
        'calinski': calinski,
        'davies': davies
    })

eval_df = pd.DataFrame(evaluations)
print("聚类评估结果:")
print(eval_df)

# 绘制评估曲线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(eval_df['k'], eval_df['silhouette'], 'bo-')
axes[0].set_xlabel('簇数 (K)')
axes[0].set_ylabel('轮廓系数')
axes[0].set_title('轮廓系数（越高越好）')
axes[0].grid(True, alpha=0.3)

axes[1].plot(eval_df['k'], eval_df['calinski'], 'go-')
axes[1].set_xlabel('簇数 (K)')
axes[1].set_ylabel('Calinski-Harabasz指数')
axes[1].set_title('Calinski-Harabasz（越高越好）')
axes[1].grid(True, alpha=0.3)

axes[2].plot(eval_df['k'], eval_df['davies'], 'ro-')
axes[2].set_xlabel('簇数 (K)')
axes[2].set_ylabel('Davies-Bouldin指数')
axes[2].set_title('Davies-Bouldin（越低越好）')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# DBSCAN 参数调优
print("\n=== DBSCAN 参数调优 ===")

eps_values = [0.2, 0.3, 0.4, 0.5]
min_samples_values = [3, 5, 7]

results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X_moons)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = np.sum(labels == -1)

        if n_clusters > 1:
            silhouette = silhouette_score(X_moons, labels)
        else:
            silhouette = 0

        results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette
        })

results_df = pd.DataFrame(results)
print("DBSCAN参数调优结果:")
print(results_df)