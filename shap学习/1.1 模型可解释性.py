# SHAP模型可解释性学习
# 主要内容：特征重要性、SHAP值计算、可视化、模型解释

# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_boston, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
import shap

# 加载数据集
print("=== 加载数据集 ===")

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"特征数量: {len(feature_names)}")
print(f"类别分布: {np.bincount(y)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练模型
print("\n=== 训练模型 ===")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 计算训练集上的准确率
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"训练集准确率: {train_acc:.4f}")
print(f"测试集准确率: {test_acc:.4f}")

# SHAP基础
print("\n=== SHAP基础 ===")

# 创建SHAP解释器
print("创建SHAP解释器...")
explainer = shap.TreeExplainer(model)
print(f"解释器类型: {type(explainer)}")

# 计算SHAP值
print("\n计算SHAP值...")
shap_values = explainer.shap_values(X_test)

print(f"SHAP值形状: {np.array(shap_values).shape}")
print(f"SHAP值示例 (第一个样本):")
print(f"  前5个特征的SHAP值: {shap_values[0][:5]}")

# SHAP值的解释
print("\n=== SHAP值解释 ===")

# 对第一个测试样本进行解释
sample_idx = 0
print(f"\n对测试样本 {sample_idx} 的解释:")
print(f"  真实标签: {y_test[sample_idx]}")
print(f"  预测概率: {model.predict_proba(X_test[sample_idx:sample_idx+1])[0]}")

# 获取该样本的SHAP值
sample_shap = shap_values[sample_idx]
print(f"\n  各特征的SHAP值 (绝对值排序):")
abs_shap = np.abs(sample_shap)
sorted_idx = np.argsort(abs_shap)[::-1]
for i in sorted_idx[:5]:
    print(f"    {feature_names[i]}: {sample_shap[i]:.4f}")

# 可视化1: 单一样本的SHAP值
print("\n=== 可视化单一样本的SHAP值 ===")

# 绘制SHAP值条形图
plt.figure(figsize=(10, 6))
plt.barh(range(10), sample_shap[sorted_idx[:10]], 
         tick_label=[feature_names[i][:20] for i in sorted_idx[:10]])
plt.xlabel('SHAP值')
plt.title(f'测试样本 {sample_idx} 的特征重要性 (SHAP值)')
plt.tight_layout()
plt.show()

# SHAP Force Plot
print("\n绘制SHAP Force Plot...")
shap.initjs()
plt.figure(figsize=(20, 4))
shap.force_plot(
    explainer.expected_value[1],  # 类别1的基准值
    shap_values[sample_idx, :],   # 该样本的SHAP值
    X_test[sample_idx, :],        # 该样本的特征值
    feature_names=feature_names,
    matplotlib=True,
    show=False
)
plt.tight_layout()
plt.show()

# 全局特征重要性
print("\n=== 全局特征重要性 ===")

# 方法1: SHAP特征重要性 (基于平均绝对SHAP值)
print("\n方法1: SHAP平均绝对值特征重要性")
shap_mean = np.abs(shap_values[:, :, 1]).mean(axis=0)  # 针对类别1
shap_importance = dict(zip(feature_names, shap_mean))
sorted_importance = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

print("Top 10 重要特征:")
for i, (name, importance) in enumerate(sorted_importance[:10], 1):
    print(f"  {i}. {name}: {importance:.4f}")

# 可视化: SHAP特征重要性
print("\n可视化SHAP特征重要性...")
plt.figure(figsize=(10, 8))
top_n = 20
names = [x[0] for x in sorted_importance[:top_n]]
values = [x[1] for x in sorted_importance[:top_n]]
plt.barh(range(top_n), values[::-1], tick_label=names[::-1])
plt.xlabel('平均|SHAP值|')
plt.title(f'SHAP特征重要性 (Top {top_n})')
plt.tight_layout()
plt.show()

# 方法2: 传统特征重要性 (基于树的分裂)
print("\n方法2: 树模型的特征重要性 (MDI)")
tree_importance = model.feature_importances_
tree_importance_dict = dict(zip(feature_names, tree_importance))
sorted_tree_importance = sorted(tree_importance_dict.items(), key=lambda x: x[1], reverse=True)

print("Top 10 重要特征:")
for i, (name, importance) in enumerate(sorted_tree_importance[:10], 1):
    print(f"  {i}. {name}: {importance:.4f}")

# 对比两种特征重要性
print("\n=== 特征重要性对比 ===")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# SHAP重要性
ax1 = axes[0]
top_n = 15
shap_names = [x[0] for x in sorted_importance[:top_n]]
shap_values_plot = [x[1] for x in sorted_importance[:top_n]]
ax1.barh(range(top_n), shap_values_plot[::-1], tick_label=shap_names[::-1])
ax1.set_xlabel('平均|SHAP值|')
ax1.set_title('SHAP特征重要性')

# 树模型重要性
ax2 = axes[1]
tree_names = [x[0] for x in sorted_tree_importance[:top_n]]
tree_values_plot = [x[1] for x in sorted_tree_importance[:top_n]]
ax2.barh(range(top_n), tree_values_plot[::-1], tick_label=tree_names[::-1])
ax2.set_xlabel('特征重要性 (MDI)')
ax2.set_title('树模型特征重要性')

plt.tight_layout()
plt.show()

# SHAP Summary Plot
print("\n=== SHAP Summary Plot ===")
print("绘制SHAP Summary Plot...")
shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names, show=False)
plt.tight_layout()
plt.show()

# SHAP Dependence Plot
print("\n=== SHAP Dependence Plot ===")
print("绘制SHAP Dependence Plot...")

# 选择最重要的特征
top_feature = sorted_importance[0][0]
top_feature_idx = list(feature_names).index(top_feature)

plt.figure(figsize=(10, 6))
shap.dependence_plot(
    top_feature_idx,
    shap_values[:, :, 1],
    X_test,
    feature_names=feature_names,
    interaction_index='auto',
    show=False
)
plt.title(f'{top_feature} 的SHAP依赖关系')
plt.tight_layout()
plt.show()

# 多特征依赖关系
print("\n绘制多特征依赖关系...")
top_3_features = [x[0] for x in sorted_importance[:3]]
top_3_indices = [list(feature_names).index(f) for f in top_3_features]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, (feature, idx) in enumerate(zip(top_3_features, top_3_indices)):
    shap.dependence_plot(
        idx,
        shap_values[:, :, 1],
        X_test,
        feature_names=feature_names,
        ax=axes[i],
        show=False
    )
    axes[i].set_title(f'{feature} 依赖关系')
plt.tight_layout()
plt.show()

# SHAP交互值
print("\n=== SHAP交互值 ===")
print("计算SHAP交互值...")
shap_interaction = explainer.shap_interaction_values(X_test[:50])  # 使用前50个样本以节省时间
print(f"SHAP交互值形状: {shap_interaction.shape}")

# 可视化交互效应
print("绘制SHAP交互效应...")
top_2_features = [x[0] for x in sorted_importance[:2]]
top_2_indices = [list(feature_names).index(f) for f in top_2_features]

plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_interaction[:, top_2_indices[0], top_2_indices[1]],
    X_test[:50],
    feature_names=feature_names,
    show=False
)
plt.title(f'{top_2_features[0]} 和 {top_2_features[1]} 的交互效应')
plt.tight_layout()
plt.show()

# 回归模型的SHAP
print("\n=== 回归模型的SHAP ===")

# 加载回归数据集
try:
    # 使用房价数据集 (sklearn 1.0+ 已移除，使用其他数据集)
    X_reg, y_reg = load_breast_cancer()['data'][:, :5], load_breast_cancer()['target']
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    # 训练回归模型
    reg_model = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_model.fit(X_reg_train, y_reg_train)
    
    # SHAP解释
    reg_explainer = shap.TreeExplainer(reg_model)
    reg_shap_values = reg_explainer.shap_values(X_reg_test)
    
    print("回归模型SHAP值计算完成")
    print(f"SHAP值形状: {reg_shap_values.shape}")
    
    # 绘制summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(reg_shap_values, X_reg_test, 
                      feature_names=data.feature_names[:5], show=False)
    plt.title('回归模型的SHAP Summary Plot')
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"回归模型示例执行出错: {e}")

# 线性模型的SHAP
print("\n=== 线性模型的SHAP ===")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 训练逻辑回归模型
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_scaled, y_train)

# 创建线性模型解释器
lr_explainer = shap.LinearExplainer(lr_model, X_train_scaled)
lr_shap_values = lr_explainer.shap_values(X_test_scaled)

print("线性模型SHAP值计算完成")
print(f"SHAP值形状: {lr_shap_values.shape}")

# 对比线性模型和树模型的SHAP值
print("\n对比线性模型和树模型的SHAP值:")
print("线性模型:")
print(f"  平均绝对SHAP值: {np.abs(lr_shap_values).mean():.4f}")
print("树模型:")
print(f"  平均绝对SHAP值: {np.abs(shap_values[:, :, 1]).mean():.4f}")

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 线性模型
shap.summary_plot(lr_shap_values, X_test_scaled, feature_names=feature_names,
                  show=False, ax=axes[0])
axes[0].set_title('线性模型 (逻辑回归)')

# 树模型
shap.summary_plot(shap_values[:, :, 1], X_test, feature_names=feature_names,
                  show=False, ax=axes[1])
axes[1].set_title('树模型 (随机森林)')

plt.tight_layout()
plt.show()

# SHAP的实际应用
print("\n=== SHAP的实践应用 ===")

# 应用1: 识别模型偏差
print("\n应用1: 识别模型偏差")
print("分析不同群体的预测差异...")

# 按真实标签分组分析
group_0_shap = shap_values[y_test == 0, :, 1].mean(axis=0)
group_1_shap = shap_values[y_test == 1, :, 1].mean(axis=0)

print("\n各类别的平均SHAP值:")
for i, name in enumerate(feature_names[:10]):
    print(f"  {name}: 类别0={group_0_shap[i]:.4f}, 类别1={group_1_shap[i]:.4f}")

# 应用2: 错误分析
print("\n应用2: 错误分析")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 找出预测错误的样本
wrong_idx = np.where(y_pred != y_test)[0]
print(f"预测错误的样本数: {len(wrong_idx)}")

if len(wrong_idx) > 0:
    # 分析错误样本
    wrong_sample_idx = wrong_idx[0]
    print(f"\n分析错误样本 {wrong_sample_idx}:")
    print(f"  真实标签: {y_test[wrong_sample_idx]}")
    print(f"  预测标签: {y_pred[wrong_sample_idx]}")
    print(f"  预测概率: {y_pred_proba[wrong_sample_idx]:.4f}")
    
    # 该样本的SHAP值
    wrong_shap = shap_values[wrong_sample_idx, :, 1]
    print(f"\n  SHAP值分析:")
    sorted_idx = np.argsort(np.abs(wrong_shap))[::-1]
    for i in sorted_idx[:5]:
        print(f"    {feature_names[i]}: {wrong_shap[i]:.4f} (特征值: {X_test[wrong_sample_idx, i]:.4f})")

print("\nSHAP模型可解释性学习完成！")
