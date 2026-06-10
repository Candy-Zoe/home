# Optuna高级优化技巧学习
# 主要内容：剪枝策略、多目标优化、自定义采样器、可视化

import optuna
import sklearn.svm
import sklearn.datasets
import sklearn.model_selection

print("=== 加载数据集 ===")
X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)

print("\n=== 定义目标函数 ===")
def objective(trial):
    C = trial.suggest_float("C", 1e-10, 1e10, log=True)
    gamma = trial.suggest_float("gamma", 1e-10, 1e10, log=True)
    kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
    
    model = sklearn.svm.SVC(C=C, gamma=gamma, kernel=kernel)
    score = sklearn.model_selection.cross_val_score(model, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    
    return accuracy

print("\n=== 标准优化 ===")
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)
print(f"最佳参数: {study.best_params}")
print(f"最佳准确率: {study.best_value:.4f}")

print("\n=== 剪枝策略 ===")
def objective_pruning(trial):
    C = trial.suggest_float("C", 1e-10, 1e10, log=True)
    gamma = trial.suggest_float("gamma", 1e-10, 1e10, log=True)
    
    for step in range(5):
        model = sklearn.svm.SVC(C=C, gamma=gamma)
        score = sklearn.model_selection.cross_val_score(model, X, y, n_jobs=-1, cv=2)
        accuracy = score.mean()
        
        trial.report(accuracy, step)
        
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return accuracy

study_prune = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study_prune.optimize(objective_pruning, n_trials=10)
print(f"剪枝后最佳准确率: {study_prune.best_value:.4f}")

print("\n=== 多目标优化 ===")
def objective_multi(trial):
    C = trial.suggest_float("C", 1e-10, 1e10, log=True)
    gamma = trial.suggest_float("gamma", 1e-10, 1e10, log=True)
    
    model = sklearn.svm.SVC(C=C, gamma=gamma)
    score = sklearn.model_selection.cross_val_score(model, X, y, n_jobs=-1, cv=3)
    accuracy = score.mean()
    complexity = C * gamma
    
    return accuracy, -complexity

study_multi = optuna.create_study(directions=["maximize", "maximize"])
study_multi.optimize(objective_multi, n_trials=10)
print(f"帕累托前沿解数量: {len(study_multi.best_trials)}")

print("\n=== 自定义采样器 ===")
class CustomSampler(optuna.samplers.BaseSampler):
    def infer_relative_search_space(self, study, trial):
        return optuna.samplers.IntersectionSearchSpace().infer_relative_search_space(study, trial)
    
    def sample_relative(self, study, trial, search_space):
        return {
            name: distribution.sample()
            for name, distribution in search_space.items()
        }
    
    def sample_independent(self, study, trial, param_name, param_distribution):
        return param_distribution.sample()

study_custom = optuna.create_study(sampler=CustomSampler())
study_custom.optimize(objective, n_trials=5)
print(f"自定义采样器最佳准确率: {study_custom.best_value:.4f}")

print("\n=== 可视化 ===")
try:
    optuna.visualization.plot_optimization_history(study)
    optuna.visualization.plot_param_importances(study)
    print("可视化图表已生成")
except:
    print("可视化需要额外依赖")

print("\n=== 超参数重要性 ===")
importance = optuna.importance.get_param_importances(study)
print("参数重要性:")
for param, importance_score in importance.items():
    print(f"  {param}: {importance_score:.4f}")