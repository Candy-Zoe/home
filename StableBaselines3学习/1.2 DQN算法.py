# Stable Baselines3 DQN算法学习
# 主要内容：DQN算法训练、评估、不同探索策略

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

print("=== 创建环境 ===")
env = make_vec_env('CartPole-v1', n_envs=1)

print("\n=== 创建DQN模型 ===")
model = DQN(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=1000,
    target_update_interval=500,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
)

print("\n=== 训练模型 ===")
model.learn(total_timesteps=10000)

print("\n=== 评估模型 ===")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

print("\n=== 清理测试文件 ===")
print("DQN训练完成")