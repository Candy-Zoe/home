# Optuna高级优化技巧学习
# 主要内容：剪枝策略、多目标优化、自定义采样器

import optuna
import sklearn
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score

print("=== 加载数据集 ===")
data = load_breast_cancer()
X, y = data.data, data.target

print("\n=== 剪枝策略 ===")
def objective_pruning(trial):
    C = trial.suggest_float('C', 1e-10, 1e10, log=True)
    gamma = trial.suggest_float('gamma', 1e-10, 1e10, log=True)
    
    model = SVC(C=C, gamma=gamma, random_state=42)
    accuracy = cross_val_score(model, X, y, cv=5).mean()
    
    trial.report(accuracy, 0)
    if trial.should_prune():
        raise optuna.TrialPruned()
    
    return accuracy

study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
study.optimize(objective_pruning, n_trials=50)
print(f"最佳准确率: {study.best_value:.4f}")

print("\n=== 多目标优化 ===")
def objective_multi(trial):
    reg_alpha = trial.suggest_float('reg_alpha', 0.0, 1.0)
    reg_lambda = trial.suggest_float('reg_lambda', 0.0, 1.0)
    
    model = sklearn.ensemble.GradientBoostingClassifier(
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=42
    )
    
    accuracy = cross_val_score(model, X, y, cv=5).mean()
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    
    return accuracy, -n_estimators

study_multi = optuna.create_study(directions=['maximize', 'minimize'])
study_multi.optimize(objective_multi, n_trials=30)

print("\n帕累托最优解:")
for i, trial in enumerate(study_multi.best_trials):
    print(f"解 {i+1}: 准确率={trial.values[0]:.4f}, -n_estimators={trial.values[1]}")

print("\n=== 自定义采样器 ===")
class CustomSampler(optuna.samplers.BaseSampler):
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.IntersectionSearchSpace().infer_relative_search_space(study, trial)
    
    def sample_relative(self, study, trial, search_space):
        return {name: distribution.sample() for name, distribution in search_space.items()}
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        return param_distribution.sample()

study_custom = optuna.create_study(sampler=CustomSampler())
study_custom.optimize(objective_pruning, n_trials=20)
print(f"自定义采样器最佳准确率: {study_custom.best_value:.4f}")