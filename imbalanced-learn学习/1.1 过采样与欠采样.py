# imbalanced-learn不平衡数据处理学习
# 主要内容：过采样、欠采样、组合采样、类别权重

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline

# 创建不平衡数据集
print("=== 创建不平衡数据集 ===")

# 方法1：使用make_classification创建不平衡数据
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],  # 90%为类别0，10%为类别1
    random_state=42
)

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")
print(f"不平衡比例: {np.bincount(y)[0] / np.bincount(y)[1]:.2f}:1")

# 可视化原始数据分布
print("\n=== 可视化原始数据分布 ===")
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='blue', label='类别0', alpha=0.5)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='red', label='类别1', alpha=0.5)
plt.title('原始不平衡数据集')
plt.legend()
plt.subplot(1, 2, 2)
plt.bar(['类别0', '类别1'], np.bincount(y), color=['blue', 'red'])
plt.title('类别分布')
plt.tight_layout()
plt.show()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print(f"训练集类别分布: {np.bincount(y_train)}")
print(f"测试集类别分布: {np.bincount(y_test)}")

# 过采样方法
print("\n=== 过采样方法 ===")

# 1. 随机过采样
print("\n1. 随机过采样 (RandomOverSampler):")
ros = RandomOverSampler(random_state=42)
X_ros, y_ros = ros.fit_resample(X_train, y_train)
print(f"  过采样后数据集大小: {X_ros.shape[0]}")
print(f"  过采样后类别分布: {np.bincount(y_ros)}")

# 2. SMOTE (Synthetic Minority Over-sampling Technique)
print("\n2. SMOTE过采样:")
smote = SMOTE(random_state=42, k_neighbors=5)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
print(f"  SMOTE后数据集大小: {X_smote.shape[0]}")
print(f"  SMOTE后类别分布: {np.bincount(y_smote)}")

# 3. ADASYN (Adaptive Synthetic Sampling)
print("\n3. ADASYN过采样:")
try:
    adasyn = ADASYN(random_state=42, n_neighbors=5)
    X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
    print(f"  ADASYN后数据集大小: {X_adasyn.shape[0]}")
    print(f"  ADASYN后类别分布: {np.bincount(y_adasyn)}")
except Exception as e:
    print(f"  ADASYN执行出错: {e}")

# 4. BorderlineSMOTE
print("\n4. BorderlineSMOTE过采样:")
try:
    borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=5)
    X_border, y_border = borderline_smote.fit_resample(X_train, y_train)
    print(f"  BorderlineSMOTE后数据集大小: {X_border.shape[0]}")
    print(f"  BorderlineSMOTE后类别分布: {np.bincount(y_border)}")
except Exception as e:
    print(f"  BorderlineSMOTE执行出错: {e}")

# 可视化SMOTE生成的数据
print("\n=== 可视化SMOTE生成的数据 ===")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 原始数据
axes[0, 0].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
                   c='blue', label='类别0', alpha=0.5)
axes[0, 0].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                   c='red', label='类别1', alpha=0.5)
axes[0, 0].set_title('原始训练数据')
axes[0, 0].legend()

# 随机过采样
axes[0, 1].scatter(X_ros[y_ros == 0][:, 0], X_ros[y_ros == 0][:, 1],
                   c='blue', label='类别0', alpha=0.5)
axes[0, 1].scatter(X_ros[y_ros == 1][:, 0], X_ros[y_ros == 1][:, 1],
                   c='red', label='类别1', alpha=0.5)
axes[0, 1].set_title('随机过采样')
axes[0, 1].legend()

# SMOTE
axes[1, 0].scatter(X_smote[y_smote == 0][:, 0], X_smote[y_smote == 0][:, 1],
                   c='blue', label='类别0', alpha=0.5)
axes[1, 0].scatter(X_smote[y_smote == 1][:, 0], X_smote[y_smote == 1][:, 1],
                   c='red', label='类别1', alpha=0.5)
axes[1, 0].set_title('SMOTE过采样')
axes[1, 0].legend()

# SMOTE vs 原始数据对比
axes[1, 1].scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1],
                   c='blue', label='原始类别0', alpha=0.3, marker='o')
axes[1, 1].scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1],
                   c='red', label='原始类别1', alpha=0.3, marker='o')
# 标记新生成的样本
new_samples_idx = smote.sample_indices_
axes[1, 1].scatter(X_smote[new_samples_idx][:, 0], X_smote[new_samples_idx][:, 1],
                   c='green', label='SMOTE生成', alpha=0.7, marker='x', s=100)
axes[1, 1].set_title('SMOTE生成的合成样本')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

# 欠采样方法
print("\n=== 欠采样方法 ===")

# 1. 随机欠采样
print("\n1. 随机欠采样 (RandomUnderSampler):")
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_train, y_train)
print(f"  欠采样后数据集大小: {X_rus.shape[0]}")
print(f"  欠采样后类别分布: {np.bincount(y_rus)}")

# 2. Tomek Links
print("\n2. Tomek Links欠采样:")
tl = TomekLinks()
X_tl, y_tl = tl.fit_resample(X_train, y_train)
print(f"  Tomek Links后数据集大小: {X_tl.shape[0]}")
print(f"  Tomek Links后类别分布: {np.bincount(y_tl)}")
print(f"  移除的样本数: {len(X_train) - len(X_tl)}")

# 3. NearMiss
print("\n3. NearMiss欠采样:")
nm = NearMiss(version=1, n_neighbors=3)
X_nm, y_nm = nm.fit_resample(X_train, y_train)
print(f"  NearMiss后数据集大小: {X_nm.shape[0]}")
print(f"  NearMiss后类别分布: {np.bincount(y_nm)}")

# 组合采样方法
print("\n=== 组合采样方法 ===")

# 1. SMOTE + ENN (Edited Nearest Neighbors)
print("\n1. SMOTEENN组合采样:")
smoteenn = SMOTEENN(random_state=42)
X_smoteenn, y_smoteenn = smoteenn.fit_resample(X_train, y_train)
print(f"  SMOTEENN后数据集大小: {X_smoteenn.shape[0]}")
print(f"  SMOTEENN后类别分布: {np.bincount(y_smoteenn)}")

# 2. SMOTE + Tomek Links
print("\n2. SMOTETomek组合采样:")
smotetomek = SMOTETomek(random_state=42)
X_smotetomek, y_smotetomek = smotetomek.fit_resample(X_train, y_train)
print(f"  SMOTETomek后数据集大小: {X_smotetomek.shape[0]}")
print(f"  SMOTETomek后类别分布: {np.bincount(y_smotetomek)}")

# 模型评估对比
print("\n=== 不同采样方法的模型评估 ===")

# 定义不同的采样方法
sampling_methods = {
    '原始数据': (X_train, y_train),
    '随机过采样': (X_ros, y_ros),
    'SMOTE': (X_smote, y_smote),
    '随机欠采样': (X_rus, y_rus),
    'SMOTEENN': (X_smoteenn, y_smoteenn),
    'SMOTETomek': (X_smotetomek, y_smotetomek)
}

# 训练和评估逻辑回归模型
print("\n逻辑回归模型在不同采样数据上的表现:")
results = {}
for name, (X_res, y_res) in sampling_methods.items():
    # 训练模型
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_res, y_res)
    
    # 在测试集上评估
    y_pred = clf.predict(X_test)
    
    # 计算准确率和召回率
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)  # 对少数类的召回率
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': acc,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }
    
    print(f"\n{name}:")
    print(f"  准确率: {acc:.4f}")
    print(f"  少数类召回率: {recall:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  F1分数: {f1:.4f}")

# 可视化评估结果
print("\n=== 评估结果可视化 ===")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 准备数据
methods = list(results.keys())
metrics = ['accuracy', 'recall', 'precision', 'f1']
titles = ['准确率 (Accuracy)', '召回率 (Recall)', '精确率 (Precision)', 'F1分数']

for ax, metric, title in zip(axes.flat, metrics, titles):
    values = [results[m][metric] for m in methods]
    bars = ax.bar(methods, values, color=['blue', 'green', 'orange', 'red', 'purple', 'cyan'])
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 使用Pipeline进行采样和训练
print("\n=== 使用Pipeline进行采样和训练 ===")

# 创建Pipeline
pipeline_lr = Pipeline([
    ('sampler', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 交叉验证评估
cv_scores = cross_val_score(pipeline_lr, X_train, y_train, cv=5, scoring='f1')
print(f"SMOTE + 逻辑回归 Pipeline 交叉验证F1分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 不同分类器的Pipeline
print("\n不同分类器 + SMOTE的Pipeline对比:")
classifiers = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=42)
}

for clf_name, clf in classifiers.items():
    pipeline = Pipeline([
        ('sampler', SMOTE(random_state=42)),
        ('classifier', clf)
    ])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='f1')
    print(f"  {clf_name}: F1={cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 混淆矩阵对比
print("\n=== 混淆矩阵对比 ===")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, (X_res, y_res)) in enumerate(sampling_methods.items()):
    # 训练模型
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_res, y_res)
    y_pred = clf.predict(X_test)
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    
    # 绘制混淆矩阵
    ax = axes[idx]
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=['类别0', '类别1'],
           yticklabels=['类别0', '类别1'],
           title=f'{name}',
           ylabel='真实标签',
           xlabel='预测标签')
    
    # 在每个格子里显示数值
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

# 真实数据集示例
print("\n=== 真实数据集示例 ===")

# 使用make_imbalance创建更复杂的不平衡数据
X_moon, y_moon = make_moons(n_samples=1000, noise=0.3, random_state=42)
# 创建不平衡版本
ratio = {0: 900, 1: 100}
X_imbalanced, y_imbalanced = make_imbalance(X_moon, y_moon, sampling_strategy=ratio)

print(f"不平衡Moon数据集:")
print(f"  数据集大小: {len(y_imbalanced)}")
print(f"  类别分布: 类别0={sum(y_imbalanced==0)}, 类别1={sum(y_imbalanced==1)}")

# 在Moon数据集上应用SMOTE
smote_moon = SMOTE(random_state=42)
X_moon_balanced, y_moon_balanced = smote_moon.fit_resample(X_imbalanced, y_imbalanced)
print(f"\nSMOTE后:")
print(f"  数据集大小: {len(y_moon_balanced)}")
print(f"  类别分布: 类别0={sum(y_moon_balanced==0)}, 类别1={sum(y_moon_balanced==1)}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 原始不平衡数据
axes[0].scatter(X_imbalanced[y_imbalanced == 0][:, 0], X_imbalanced[y_imbalanced == 0][:, 1],
                c='blue', label='类别0', alpha=0.5)
axes[0].scatter(X_imbalanced[y_imbalanced == 1][:, 0], X_imbalanced[y_imbalanced == 1][:, 1],
                c='red', label='类别1', alpha=0.8)
axes[0].set_title('原始不平衡Moon数据')
axes[0].legend()

# SMOTE后
axes[1].scatter(X_moon_balanced[y_moon_balanced == 0][:, 0], X_moon_balanced[y_moon_balanced == 0][:, 1],
                c='blue', label='类别0', alpha=0.5)
axes[1].scatter(X_moon_balanced[y_moon_balanced == 1][:, 0], X_moon_balanced[y_moon_balanced == 1][:, 1],
                c='red', label='类别1', alpha=0.5)
axes[1].set_title('SMOTE处理后的Moon数据')
axes[1].legend()

plt.tight_layout()
plt.show()

print("\nimbalanced-learn不平衡数据处理学习完成！")
