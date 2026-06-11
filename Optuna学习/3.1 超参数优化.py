# Optuna超参数优化学习
# 主要内容：Optuna基础、目标函数、搜索空间、剪枝、集成

# 导入必要的库
import optuna
from optuna.samplers import TPESampler, RandomSampler, GridSampler
from optuna.pruners import MedianPruner, HyperbandPruner
from optuna.integration import OptunaSearchCV
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Optuna基础
print("=== Optuna基础 ===")
print(f"Optuna版本: {optuna.__version__}")

# 定义一个简单的优化问题
print("\n=== 简单的优化问题 ===")

# 定义目标函数
def objective(trial):
    """简单的目标函数：最小化 (x - 3)^2"""
    x = trial.suggest_float('x', -10, 10)
    return (x - 3) ** 2

# 创建研究
study = optuna.create_study(
    direction='minimize',  # 最小化目标函数
    sampler=TPESampler(seed=42)
)

# 运行优化
print("优化 y = (x - 3)^2:")
study.optimize(objective, n_trials=100, show_progress_bar=False)

print(f"最优值: x = {study.best_params['x']:.4f}")
print(f"最小值: {study.best_value:.6f}")

# 可视化优化过程
plt.figure(figsize=(12, 4))

# 绘制目标函数
x = np.linspace(-10, 10, 100)
y = (x - 3) ** 2
plt.subplot(1, 2, 1)
plt.plot(x, y, 'b-', linewidth=2, label='目标函数')
plt.axvline(x=study.best_params['x'], color='r', linestyle='--', label=f'最优x={study.best_params["x"]:.4f}')
plt.xlabel('x')
plt.ylabel('y')
plt.title('目标函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 绘制优化历史
plt.subplot(1, 2, 2)
values = [trial.value for trial in study.trials]
plt.plot(values, 'b-', alpha=0.6)
plt.xlabel('试验次数')
plt.ylabel('目标值')
plt.title('优化历史')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 多参数优化
print("\n=== 多参数优化 ===")

def objective_multi(trial):
    """多参数目标函数"""
    x = trial.suggest_float('x', -10, 10)
    y = trial.suggest_float('y', -10, 10)
    z = trial.suggest_float('z', -5, 5)
    
    # Rosenbrock函数
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2 + (1 - z) ** 2

study_multi = optuna.create_study(
    direction='minimize',
    sampler=TPESampler(seed=42)
)

study_multi.optimize(objective_multi, n_trials=200, show_progress_bar=False)

print(f"最优参数:")
for key, value in study_multi.best_params.items():
    print(f"  {key}: {value:.4f}")
print(f"最小值: {study_multi.best_value:.6f}")

# 机器学习模型优化
print("\n=== 机器学习模型优化 ===")

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RandomForest超参数优化
print("\nRandomForest超参数优化:")

def objective_rf(trial):
    """RandomForest目标函数"""
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        n_jobs=-1
    )
    
    # 交叉验证
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

# 创建研究
study_rf = optuna.create_study(
    direction='maximize',  # 最大化准确率
    sampler=TPESampler(seed=42)
)

study_rf.optimize(objective_rf, n_trials=50, show_progress_bar=False)

print(f"最优参数: {study_rf.best_params}")
print(f"最佳交叉验证分数: {study_rf.best_value:.4f}")

# 使用最优参数训练最终模型
best_rf = RandomForestClassifier(
    **study_rf.best_params,
    random_state=42,
    n_jobs=-1
)
best_rf.fit(X_train, y_train)
test_score = best_rf.score(X_test, y_test)
print(f"测试集分数: {test_score:.4f}")

# 绘制参数重要性
optuna.visualization.plot_param_importances(study_rf).show()

# GradientBoosting优化
print("\nGradientBoosting超参数优化:")

def objective_gb(trial):
    """GradientBoosting目标函数"""
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

study_gb = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study_gb.optimize(objective_gb, n_trials=30, show_progress_bar=False)

print(f"最优参数: {study_gb.best_params}")
print(f"最佳交叉验证分数: {study_gb.best_value:.4f}")

# SVM优化
print("\nSVM超参数优化:")

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

def objective_svm(trial):
    """SVM目标函数"""
    C = trial.suggest_float('C', 0.01, 100, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'rbf', 'poly'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    model = SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        random_state=42
    )
    
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
    return scores.mean()

study_svm = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study_svm.optimize(objective_svm, n_trials=30, show_progress_bar=False)

print(f"最优参数: {study_svm.best_params}")
print(f"最佳交叉验证分数: {study_svm.best_value:.4f}")

# 模型对比
print("\n=== 模型对比 ===")

models = {
    'RandomForest': (study_rf.best_value, test_score),
    'GradientBoosting': (study_gb.best_value, GradientBoostingClassifier(
        **study_gb.best_params, random_state=42
    ).fit(X_train, y_train).score(X_test, y_test)),
    'SVM': (study_svm.best_value, SVC(
        **study_svm.best_params, random_state=42
    ).fit(X_train_scaled, y_train).score(X_test_scaled, y_test))
}

print("模型性能对比:")
for name, (cv_score, test_score) in models.items():
    print(f"  {name}: CV分数={cv_score:.4f}, 测试分数={test_score:.4f}")

# 可视化
plt.figure(figsize=(10, 5))
names = list(models.keys())
cv_scores = [models[n][0] for n in names]
test_scores = [models[n][1] for n in names]

x = np.arange(len(names))
width = 0.35

plt.bar(x - width/2, cv_scores, width, label='交叉验证分数')
plt.bar(x + width/2, test_scores, width, label='测试分数')

plt.xlabel('模型')
plt.ylabel('准确率')
plt.title('模型对比')
plt.xticks(x, names)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# Optuna可视化
print("\n=== Optuna可视化 ===")

# 优化历史
fig1 = optuna.visualization.plot_optimization_history(study_rf)
fig1.show()

# 参数重要性
fig2 = optuna.visualization.plot_param_importances(study_rf)
fig2.show()

# 并行坐标图
fig3 = optuna.visualization.plot_parallel_coordinate(study_rf)
fig3.show()

# 切片图
fig4 = optuna.visualization.plot_slice(study_rf)
fig4.show()

# 等高线图
fig5 = optuna.visualization.plot_contour(study_rf, params=['n_estimators', 'max_depth'])
fig5.show()

# 剪枝
print("\n=== 剪枝（早停） ===")

def objective_with_pruning(trial):
    """带剪枝的目标函数"""
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    
    # 交叉验证
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    
    # 报告中间结果用于剪枝
    trial.report(np.mean(scores), step=0)
    
    # 检查是否应该剪枝
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return np.mean(scores)

# 使用剪枝器
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)
study_pruning = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=pruner
)

study_pruning.optimize(objective_with_pruning, n_trials=50, show_progress_bar=False)

print(f"使用剪枝后的最优分数: {study_pruning.best_value:.4f}")
print(f"完成的试验数: {len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
print(f"被剪枝的试验数: {len([t for t in study_pruning.trials if t.state == optuna.trial.TrialState.PRUNED])}")

# 搜索空间定义
print("\n=== 搜索空间定义 ===")

# 使用条件参数
def objective_conditional(trial):
    """条件参数示例"""
    classifier_type = trial.suggest_categorical('classifier', ['rf', 'gb', 'svm'])
    
    if classifier_type == 'rf':
        n_estimators = trial.suggest_int('rf_n_estimators', 10, 200)
        max_depth = trial.suggest_int('rf_max_depth', 2, 20)
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
    elif classifier_type == 'gb':
        n_estimators = trial.suggest_int('gb_n_estimators', 10, 200)
        learning_rate = trial.suggest_float('gb_learning_rate', 0.01, 0.3, log=True)
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42
        )
    else:  # svm
        C = trial.suggest_float('svm_C', 0.01, 100, log=True)
        model = SVC(C=C, random_state=42)
    
    scores = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy')
    return scores.mean()

study_conditional = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study_conditional.optimize(objective_conditional, n_trials=30, show_progress_bar=False)

print(f"条件搜索最优分类器: {study_conditional.best_params['classifier']}")
print(f"最优分数: {study_conditional.best_value:.4f}")
print(f"最优参数: {study_conditional.best_params}")

# 多目标优化
print("\n=== 多目标优化 ===")

def objective_multi_objective(trial):
    """多目标：同时最大化准确率，最小化模型复杂度"""
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1
    )
    
    # 准确率
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    accuracy = np.mean(scores)
    
    # 模型复杂度（使用n_estimators * max_depth作为代理）
    complexity = n_estimators * max_depth
    
    return accuracy, complexity

# 多目标研究
study_multi_obj = optuna.create_study(
    directions=['maximize', 'minimize'],  # 最大化准确率，最小化复杂度
    sampler=TPESampler(seed=42)
)

study_multi_obj.optimize(objective_multi_objective, n_trials=50, show_progress_bar=False)

print(f"Pareto前沿解数量: {len(study_multi_obj.best_trials)}")
print("\nPareto前沿解:")
for trial in study_multi_obj.best_trials[:5]:
    print(f"  准确率: {trial.values[0]:.4f}, 复杂度: {trial.values[1]:.0f}")
    print(f"  参数: {trial.params}")

# 可视化Pareto前沿
optuna.visualization.plot_pareto_front(study_multi_obj).show()

# 与sklearn集成
print("\n=== 与sklearn集成 ===")

# 使用OptunaSearchCV
param_distributions = {
    'n_estimators': optuna.distributions.IntDistribution(10, 200),
    'max_depth': optuna.distributions.IntDistribution(2, 20),
    'min_samples_split': optuna.distributions.IntDistribution(2, 20),
    'min_samples_leaf': optuna.distributions.IntDistribution(1, 10),
}

optuna_search = OptunaSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=param_distributions,
    n_trials=30,
    direction='maximize',
    cv=3,
    sampler=TPESampler(seed=42),
    pruner=MedianPruner()
)

optuna_search.fit(X_train, y_train)

print(f"最优参数: {optuna_search.best_params_}")
print(f"交叉验证分数: {optuna_search.best_score_:.4f}")
print(f"测试集分数: {optuna_search.score(X_test, y_test):.4f}")

# 保存和加载研究
print("\n=== 保存和加载研究 ===")

# 保存研究
study_rf.set_user_attr('best_accuracy', study_rf.best_value)
optuna.storages.JournalStorage(
    optuna.storages.JournalFileStorage('./optuna_study.log')
)

# 加载研究
study_loaded = optuna.create_study(
    storage='sqlite:///example.db',
    sampler=TPESampler(seed=42),
    study_name='loaded_study'
)

print("研究保存和加载功能已演示")

# 清理
import os
if os.path.exists('example.db'):
    os.remove('example.db')
    print("临时数据库已清理")

# 回调函数
print("\n=== 回调函数 ===")

def logging_callback(study, trial):
    """日志回调"""
    print(f"Trial {trial.number}: value={trial.value:.4f}, params={trial.params}")

study_callback = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study_callback.optimize(
    objective_rf,
    n_trials=10,
    callbacks=[logging_callback],
    show_progress_bar=False
)

# 约束条件
print("\n=== 约束条件 ===")

def objective_with_constraint(trial):
    """带约束的目标函数"""
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    
    # 约束：max_depth * min_samples_split <= 100
    if max_depth * min_samples_split > 100:
        raise optuna.exceptions.TrialPruned()
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42,
        n_jobs=-1
    )
    
    scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
    return scores.mean()

study_constrained = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42)
)

study_constrained.optimize(objective_with_constraint, n_trials=50, show_progress_bar=False)
print(f"带约束的最优分数: {study_constrained.best_value:.4f}")

# 总结
print("\n=== Optuna超参数优化学习总结 ===")
print("1. Optuna基础和目标函数")
print("2. 多参数优化")
print("3. 机器学习模型超参数优化")
print("4. 多种采样器（ TPE、Random、Grid）")
print("5. 剪枝机制（早停）")
print("6. 条件参数")
print("7. 多目标优化")
print("8. Optuna可视化")
print("9. 与sklearn集成")
print("10. 研究保存和加载")
print("11. 回调函数")
print("12. 约束条件")

print("\nOptuna超参数优化学习完成！")
