# UMAP降维可视化学习
# 主要内容：降维、聚类可视化、数据探索、流形学习

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_digits, load_wine, make_blobs, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# 创建示例数据
print("=== 创建示例数据 ===")

# 加载鸢尾花数据集
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
feature_names_iris = iris.feature_names
print(f"鸢尾花数据集: X={X_iris.shape}, y={y_iris.shape}")
print(f"类别: {np.unique(y_iris)}")

# 加载手写数字数据集
digits = load_digits()
X_digits = digits.data
y_digits = digits.target
print(f"手写数字数据集: X={X_digits.shape}, y={y_digits.shape}")

# 创建自定义数据集
X_blobs, y_blobs = make_blobs(n_samples=500, centers=4, n_features=10, random_state=42)
print(f"自定义聚类数据集: X={X_blobs.shape}, y={y_blobs.shape}")

# 基本UMAP降维
print("\n=== UMAP基本降维 ===")

# 鸢尾花数据集降维
reducer = umap.UMAP(random_state=42)
embedding_iris = reducer.fit_transform(X_iris)

print(f"UMAP降维后的形状: {embedding_iris.shape}")
print(f"原始维度: {X_iris.shape[1]}, 降维后维度: {embedding_iris.shape[1]}")

# 可视化UMAP结果
print("\n=== 可视化UMAP降维结果 ===")
plt.figure(figsize=(10, 8))
plt.scatter(embedding_iris[:, 0], embedding_iris[:, 1], 
           c=y_iris, cmap='viridis', s=50, alpha=0.8)
plt.colorbar(label='类别')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('UMAP降维 - 鸢尾花数据集')
plt.tight_layout()
plt.show()

# UMAP vs PCA vs t-SNE对比
print("\n=== UMAP vs PCA vs t-SNE对比 ===")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"PCA解释方差比例: {pca.explained_variance_ratio_.sum():.4f}")

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# UMAP降维
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.8)
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_title(f'PCA (方差解释: {pca.explained_variance_ratio_.sum():.2%})')

# t-SNE
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.8)
axes[1].set_xlabel('t-SNE维度1')
axes[1].set_ylabel('t-SNE维度2')
axes[1].set_title('t-SNE')

# UMAP
axes[2].scatter(X_umap[:, 0], X_umap[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.8)
axes[2].set_xlabel('UMAP维度1')
axes[2].set_ylabel('UMAP维度2')
axes[2].set_title('UMAP')

plt.tight_layout()
plt.show()

# UMAP参数调优
print("\n=== UMAP参数调优 ===")

# 1. n_neighbors参数
print("\n1. n_neighbors参数 (邻居数量)")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
n_neighbors_values = [5, 15, 30, 50]

for ax, n_neighbors in zip(axes.flat, n_neighbors_values):
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
    embedding = reducer.fit_transform(X_iris)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')
    ax.set_title(f'n_neighbors={n_neighbors}')

plt.suptitle('不同n_neighbors值的UMAP结果', fontsize=14)
plt.tight_layout()
plt.show()

# 2. min_dist参数
print("\n2. min_dist参数 (最小距离)")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
min_dist_values = [0.0, 0.1, 0.3, 0.5]

for ax, min_dist in zip(axes.flat, min_dist_values):
    reducer = umap.UMAP(n_components=2, min_dist=min_dist, random_state=42)
    embedding = reducer.fit_transform(X_iris)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')
    ax.set_title(f'min_dist={min_dist}')

plt.suptitle('不同min_dist值的UMAP结果', fontsize=14)
plt.tight_layout()
plt.show()

# 3. metric参数
print("\n3. metric参数 (距离度量)")
metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for ax, metric in zip(axes.flat, metrics):
    reducer = umap.UMAP(n_components=2, metric=metric, random_state=42)
    embedding = reducer.fit_transform(X_iris)
    ax.scatter(embedding[:, 0], embedding[:, 1], c=y_iris, cmap='viridis', s=50, alpha=0.8)
    ax.set_xlabel('维度1')
    ax.set_ylabel('维度2')
    ax.set_title(f'metric={metric}')

plt.suptitle('不同距离度量的UMAP结果', fontsize=14)
plt.tight_layout()
plt.show()

# UMAP用于聚类可视化
print("\n=== UMAP聚类可视化 ===")

# 聚类数据集
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
embedding_blobs = reducer.fit_transform(X_blobs)

plt.figure(figsize=(10, 8))
plt.scatter(embedding_blobs[:, 0], embedding_blobs[:, 1], 
           c=y_blobs, cmap='tab10', s=50, alpha=0.8)
plt.colorbar(label='聚类')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('UMAP聚类可视化 - 自定义聚类数据')
plt.tight_layout()
plt.show()

# 高维数据集示例 - 手写数字
print("\n=== 手写数字数据集UMAP ===")
print("对手写数字数据集进行UMAP降维...")

# UMAP降维
reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
embedding_digits = reducer.fit_transform(X_digits)

# 可视化
plt.figure(figsize=(12, 10))
scatter = plt.scatter(embedding_digits[:, 0], embedding_digits[:, 1], 
                     c=y_digits, cmap='tab10', s=10, alpha=0.6)
plt.colorbar(scatter, label='数字类别')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('UMAP降维 - 手写数字数据集')
plt.tight_layout()
plt.show()

# UMAP用于异常检测
print("\n=== UMAP用于异常检测 ===")

# 创建包含异常点的数据集
np.random.seed(42)
normal_data = np.random.randn(100, 5)
outliers = np.random.randn(20, 5) * 3 + 10  # 异常点在远离中心的位置
X_combined = np.vstack([normal_data, outliers])
y_combined = np.array([0] * 100 + [1] * 20)  # 0为正常，1为异常

# UMAP降维
reducer = umap.UMAP(n_components=2, random_state=42)
embedding_combined = reducer.fit_transform(X_combined)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(embedding_combined[y_combined == 0, 0], 
           embedding_combined[y_combined == 0, 1],
           c='blue', label='正常', s=50, alpha=0.6)
plt.scatter(embedding_combined[y_combined == 1, 0], 
           embedding_combined[y_combined == 1, 1],
           c='red', label='异常', s=100, marker='x')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('UMAP异常检测可视化')
plt.legend()
plt.tight_layout()
plt.show()

# UMAP用于分类器决策边界可视化
print("\n=== UMAP分类器决策边界可视化 ===")

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# 训练多个分类器
classifiers = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'SVM': SVC(kernel='rbf', gamma='scale'),
    'RandomForest': RandomForestClassifier(n_estimators=50, random_state=42)
}

fig, axes = plt.subplots(2, 2, figsize=(14, 14))

# 原始数据 UMAP
axes[0, 0].scatter(embedding_iris[:, 0], embedding_iris[:, 1], 
                   c=y_iris, cmap='viridis', s=50, alpha=0.8)
axes[0, 0].set_title('原始数据 UMAP')
axes[0, 0].set_xlabel('UMAP维度1')
axes[0, 0].set_ylabel('UMAP维度2')

# 训练分类器并在UMAP空间可视化决策边界
for ax, (name, clf) in zip(axes.flat[1:], classifiers.items()):
    # 训练分类器
    clf.fit(embedding_iris, y_iris)
    
    # 创建网格
    x_min, x_max = embedding_iris[:, 0].min() - 1, embedding_iris[:, 0].max() + 1
    y_min, y_max = embedding_iris[:, 1].min() - 1, embedding_iris[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    ax.scatter(embedding_iris[:, 0], embedding_iris[:, 1], 
              c=y_iris, cmap='viridis', s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.set_title(f'{name} 决策边界')
    ax.set_xlabel('UMAP维度1')
    ax.set_ylabel('UMAP维度2')

plt.suptitle('UMAP空间中的分类器决策边界', fontsize=14)
plt.tight_layout()
plt.show()

# UMAP用于时间序列分析
print("\n=== UMAP用于数据预处理 ===")

# UMAP作为特征提取器
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier

# 创建Pipeline: UMAP降维 + 分类器
pipeline = Pipeline([
    ('umap', umap.UMAP(n_components=3, random_state=42)),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# 评估Pipeline
from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X_iris, y_iris, cv=5)
print(f"UMAP特征提取 + 分类器 交叉验证准确率: {scores.mean():.4f} (+/- {scores.std():.4f})")

# 对比原始数据
scores_original = cross_val_score(
    GradientBoostingClassifier(n_estimators=100, random_state=42),
    X_iris, y_iris, cv=5
)
print(f"原始数据分类器 交叉验证准确率: {scores_original.mean():.4f} (+/- {scores_original.std():.4f})")

# UMAP保存和加载
print("\n=== UMAP模型保存和加载 ===")

import pickle

# 保存UMAP模型
reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(X_iris[:100])  # 使用子集训练

with open('umap_model.pkl', 'wb') as f:
    pickle.dump(reducer, f)
print("UMAP模型已保存为 umap_model.pkl")

# 加载UMAP模型
with open('umap_model.pkl', 'rb') as f:
    loaded_reducer = pickle.load(f)

# 使用加载的模型进行预测
new_embedding = loaded_reducer.transform(X_iris[100:150])
print(f"使用加载模型处理新数据: {new_embedding.shape}")

# 清理
import os
if os.path.exists('umap_model.pkl'):
    os.remove('umap_model.pkl')
    print("已删除模型文件")

# UMAP在文本数据上的应用
print("\n=== UMAP在文本数据上的应用 ===")

from sklearn.feature_extraction.text import TfidfVectorizer

# 示例文本数据
texts = [
    "机器学习是人工智能的分支",
    "深度学习使用神经网络",
    "自然语言处理处理文本",
    "计算机视觉处理图像",
    "强化学习通过奖励学习",
    "监督学习需要标注数据",
    "无监督学习发现数据模式",
    "图像识别是计算机视觉任务"
]

# TF-IDF向量化
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(texts)

print(f"TF-IDF矩阵形状: {X_text.shape}")

# UMAP降维
reducer = umap.UMAP(n_components=2, metric='cosine', random_state=42)
embedding_text = reducer.fit_transform(X_text.toarray())

# 可视化
plt.figure(figsize=(10, 8))
for i, text in enumerate(texts):
    plt.scatter(embedding_text[i, 0], embedding_text[i, 1], s=100, c='blue')
    plt.annotate(text[:10], (embedding_text[i, 0], embedding_text[i, 1]),
                fontsize=8, alpha=0.7)
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('UMAP文本聚类可视化')
plt.tight_layout()
plt.show()

# 高级UMAP功能
print("\n=== 高级UMAP功能 ===")

# 1. 3D UMAP
print("\n1. 3D UMAP降维")
reducer_3d = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
embedding_3d = reducer_3d.fit_transform(X_iris)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2],
                    c=y_iris, cmap='viridis', s=50, alpha=0.8)
ax.set_xlabel('UMAP维度1')
ax.set_ylabel('UMAP维度2')
ax.set_zlabel('UMAP维度3')
ax.set_title('3D UMAP降维 - 鸢尾花数据集')
plt.colorbar(scatter, label='类别')
plt.tight_layout()
plt.show()

# 2. Supervised UMAP
print("\n2. Supervised UMAP (监督式UMAP)")
reducer_supervised = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
embedding_supervised = reducer_supervised.fit_transform(X_iris, y=y_iris)

plt.figure(figsize=(10, 8))
plt.scatter(embedding_supervised[:, 0], embedding_supervised[:, 1], 
           c=y_iris, cmap='viridis', s=50, alpha=0.8)
plt.colorbar(label='类别')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('Supervised UMAP (使用标签信息)')
plt.tight_layout()
plt.show()

# 3. Semisupervised UMAP
print("\n3. Semisupervised UMAP (半监督式UMAP)")
# 使用部分标签
y_partial = y_iris.copy()
y_partial[50:] = -1  # 部分样本没有标签

reducer_semi = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
embedding_semi = reducer_semi.fit_transform(X_iris, y=y_partial)

plt.figure(figsize=(10, 8))
plt.scatter(embedding_semi[:, 0], embedding_semi[:, 1], 
           c=y_partial, cmap='viridis', s=50, alpha=0.8)
plt.colorbar(label='类别 (-1表示无标签)')
plt.xlabel('UMAP维度1')
plt.ylabel('UMAP维度2')
plt.title('Semisupervised UMAP (部分有标签)')
plt.tight_layout()
plt.show()

print("\nUMAP降维可视化学习完成！")
