# Optuna超参数优化学习
# 主要内容：目标函数定义、搜索空间、优化算法、学习曲线可视化

# 导入Optuna库
import optuna
from optuna.samplers import TPESampler
import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# 设置随机种子
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(42)

# 加载数据集
print("=== 加载数据集 ===")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"数据集形状: X={X.shape}, y={y.shape}")
print(f"类别分布: {np.bincount(y)}")

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 定义目标函数
print("\n=== 定义目标函数 ===")

def objective_rf(trial):
    """随机森林目标函数"""

    # 定义超参数搜索空间
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])

    # 创建模型
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )

    # 交叉验证评估
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    accuracy = score.mean()

    return accuracy

# 执行优化
print("\n=== 执行随机森林超参数优化 ===")

# 创建study对象
study_rf = optuna.create_study(
    direction='maximize',  # 最大化准确率
    sampler=TPESampler(seed=42)  # 使用TPE采样器
)

# 执行优化
study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=False)

print(f"最佳试验:")
print(f"  试验编号: {study_rf.best_trial.number}")
print(f"  最佳准确率: {study_rf.best_value:.4f}")
print(f"  最佳参数: {study_rf.best_params}")

# 优化结果可视化
print("\n=== 优化结果可视化 ===")

# 绘制优化历史
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# 优化历史
axes[0].plot([t.number for t in study_rf.trials],
             [t.value for t in study_rf.trials], 'o')
axes[0].axhline(y=study_rf.best_value, color='r', linestyle='--', label='最佳值')
axes[0].set_xlabel('试验编号')
axes[0].set_ylabel('准确率')
axes[0].set_title('优化历史')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 参数重要性
fig2 = optuna.visualization.matplotlib.plot_param_importances(study_rf)
plt.tight_layout()
plt.show()

# 使用最佳参数训练最终模型
print("\n=== 使用最佳参数训练最终模型 ===")

best_params_rf = study_rf.best_params
final_model_rf = RandomForestClassifier(
    **best_params_rf,
    random_state=42,
    n_jobs=-1
)
final_model_rf.fit(X_train, y_train)

train_score = final_model_rf.score(X_train, y_train)
val_score = final_model_rf.score(X_val, y_val)

print(f"训练集准确率: {train_score:.4f}")
print(f"验证集准确率: {val_score:.4f}")

# 多算法对比优化
print("\n=== 多算法对比优化 ===")

def objective_svm(trial):
    """SVM目标函数"""

    # 定义超参数搜索空间
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])

    # 创建模型
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=42
    )

    # 交叉验证评估
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    accuracy = score.mean()

    return accuracy

def objective_lr(trial):
    """逻辑回归目标函数"""

    # 定义超参数搜索空间
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    # 创建模型
    model = LogisticRegression(
        C=C,
        solver=solver,
        max_iter=max_iter,
        random_state=42
    )

    # 交叉验证评估
    score = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    accuracy = score.mean()

    return accuracy

# 创建多个study
study_svm = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
study_lr = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))

# 执行优化
print("正在优化SVM...")
study_svm.optimize(objective_svm, n_trials=30, show_progress_bar=False)

print("正在优化逻辑回归...")
study_lr.optimize(objective_lr, n_trials=30, show_progress_bar=False)

# 对比结果
print("\n=== 算法对比 ===")

results = {
    '随机森林': study_rf.best_value,
    'SVM': study_svm.best_value,
    '逻辑回归': study_lr.best_value
}

for name, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {score:.4f}")

# 使用Optuna的Dashboard（需要额外安装）
print("\n=== 学习曲线分析 ===")

# 分析单个参数对性能的影响
study_analysis = optuna.create_study(direction='maximize')
study_analysis.optimize(objective_rf, n_trials=50, show_progress_bar=False)

# 绘制学习曲线
fig = optuna.visualization.matplotlib.plot_optimization_history(study_analysis)
plt.title('随机森林优化历史')
plt.tight_layout()
plt.show()

# 绘制参数关系热力图
try:
    fig = optuna.visualization.matplotlib.plot_slice(study_analysis)
    plt.title('参数切片分析')
    plt.tight_layout()
    plt.show()
except:
    print("参数切片分析需要更多试验数据")

# 使用Optuna的Pruner进行早停
print("\n=== 使用MedianPruner进行早停 ===")

# 创建带有早停的study
study_pruned = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=optuna.pruners.MedianPruner()  # 使用中位数剪枝器
)

def objective_with_pruning(trial):
    """带有早停的目标函数"""

    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 1, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )

    # 使用交叉验证
    for step in range(3):  # 最多3步
        score = cross_val_score(model, X_train, y_train, cv=2, scoring='accuracy')
        mean_score = score.mean()

        # 报告中间结果以支持早停
        trial.report(mean_score, step)

        # 如果当前步骤的性能低于中位数，停止试验
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return mean_score

study_pruned.optimize(objective_with_pruning, n_trials=30, show_progress_bar=False)
print(f"早停优化 - 最佳准确率: {study_pruned.best_value:.4f}")
print(f"完成的试验数: {len([t for t in study_pruned.trials if t.state == optuna.trial.TrialState.COMPLETE])}")

print("\n学习完成！")