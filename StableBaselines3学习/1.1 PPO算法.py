# Stable Baselines3 PPO算法学习
# 主要内容：PPO算法训练、评估、保存加载

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy

print("=== 创建向量环境 ===")
env = make_vec_env('CartPole-v1', n_envs=4)
print(f"环境数量: {env.num_envs}")

print("\n=== 创建PPO模型 ===")
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
)

print("\n=== 训练模型 ===")
model.learn(total_timesteps=10000)

print("\n=== 保存模型 ===")
model.save("ppo_cartpole")
print("模型已保存")

print("\n=== 加载模型 ===")
del model
model = PPO.load("ppo_cartpole")
print("模型已加载")

print("\n=== 评估模型 ===")
mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

print("\n=== 可视化训练 ===")
env = model.get_env()
obs = env.reset()
total_reward = 0

for _ in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    total_reward += rewards[0]
    if dones[0]:
        break

print(f"测试奖励: {total_reward}")

print("\n=== 清理测试文件 ===")
import os
if os.path.exists('ppo_cartpole.zip'):
    os.remove('ppo_cartpole.zip')
if os.path.exists('ppo_cartpole.pkl'):
    os.remove('ppo_cartpole.pkl')
print("已删除测试文件")