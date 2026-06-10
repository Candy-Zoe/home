# Stable Baselines3深度强化学习进阶学习
# 主要内容：TD3算法、PPO进阶、DDPG算法、多环境训练

import gymnasium as gym
from stable_baselines3 import DDPG, TD3, PPO, SAC
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建环境 ===")
env = gym.make("Pendulum-v1")
eval_env = gym.make("Pendulum-v1")

print(f"状态空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

print("\n=== DDPG算法 ===")
ddpg_model = DDPG(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=100000,
    learning_starts=100,
    batch_size=64,
    tau=0.005,
    gamma=0.99,
    verbose=1
)
print("DDPG模型已创建")

print("\n=== TD3算法 ===")
td3_model = TD3(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_delay=2,
    verbose=1
)
print("TD3模型已创建")

print("\n=== SAC算法 ===")
sac_model = SAC(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    buffer_size=100000,
    learning_starts=100,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    ent_coef='auto',
    verbose=1
)
print("SAC模型已创建")

print("\n=== PPO算法 ===")
ppo_model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1
)
print("PPO模型已创建")

print("\n=== 向量化环境 ===")
vec_env = make_vec_env("CartPole-v1", n_envs=4)

print(f"向量化环境数量: {vec_env.num_envs}")

print("\n=== 回调函数 ===")
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",
    log_path="./logs/eval",
    eval_freq=500,
    deterministic=True,
    render=False
)

checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./logs/checkpoints",
    name_prefix="rl_model"
)
print("回调函数已创建")

print("\n=== 训练模型 ===")
print("开始训练PPO模型(短时间演示)...")
ppo_model = PPO("MlpPolicy", "CartPole-v1", verbose=0)
ppo_model.learn(total_timesteps=5000)
print("训练完成")

print("\n=== 评估模型 ===")
from stable_baselines3.common.evaluation import evaluate_policy

mean_reward, std_reward = evaluate_policy(
    ppo_model,
    eval_env,
    n_eval_episodes=10,
    deterministic=True
)
print(f"平均奖励: {mean_reward:.2f} +/- {std_reward:.2f}")

print("\n=== 训练曲线可视化 ===")
from stable_baselines3.common.monitor import Monitor

env = Monitor(env, "monitor.csv")

print("\n=== 保存和加载模型 ===")
ppo_model.save("ppo_pendulum")
print("模型已保存")

loaded_model = PPO.load("ppo_pendulum")
print("模型已加载")

print("\n=== 继续训练 ===")
loaded_model.learn(total_timesteps=1000, reset_num_timesteps=False)
print("继续训练完成")

print("\n=== 预测与交互 ===")
obs, info = env.reset()
for i in range(100):
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
print("交互完成")

print("\n=== 清理测试文件 ===")
import shutil
import os
for path in ["./logs", "ppo_pendulum.zip", "monitor.csv"]:
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
        print(f"已删除 {path}")