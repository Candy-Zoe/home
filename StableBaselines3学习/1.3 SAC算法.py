# Stable Baselines3 SAC算法学习
# 主要内容：SAC算法训练、连续动作空间处理

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

print("=== 创建连续动作环境 ===")
env = make_vec_env('Pendulum-v1', n_envs=1)
print(f"动作空间类型: {env.action_space}")
print(f"动作空间范围: low={env.action_space.low}, high={env.action_space.high}")

print("\n=== 创建SAC模型 ===")
model = SAC(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    buffer_size=1000000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
)

print("\n=== 训练模型 ===")
model.learn(total_timesteps=20000)

print("\n=== 评估模型 ===")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

print("\n=== 测试连续动作输出 ===")
obs = env.reset()
action, _states = model.predict(obs)
print(f"动作输出: {action}")
print(f"动作形状: {action.shape}")

print("\n=== 清理测试文件 ===")
print("SAC训练完成")