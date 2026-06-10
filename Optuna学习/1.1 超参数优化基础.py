# Optuna超参数优化基础学习
# 主要内容：Optuna基础用法、搜索空间定义、优化目标

import optuna
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score

print("=== 加载数据集 ===")
iris = load_iris()
X, y = iris.data, iris.target

print("\n=== 定义优化目标函数 ===")
def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 10, 200)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    accuracy = cross_val_score(model, X, y, cv=5).mean()
    return accuracy

print("\n=== 创建优化器 ===")
study = optuna.create_study(direction='maximize')

print("\n=== 运行优化 ===")
study.optimize(objective, n_trials=50)

print("\n=== 优化结果 ===")
print(f"最佳准确率: {study.best_value:.4f}")
print(f"最佳参数: {study.best_params}")

print("\n=== 可视化优化过程 ===")
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()

print("\n=== 超参数重要性 ===")
importance = optuna.importance.get_param_importances(study)
print(f"参数重要性: {importance}")