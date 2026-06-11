# StableBaselines3 PPO算法学习
# 主要内容：PPO算法基础、环境交互、模型训练

# 导入必要的库
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# 创建环境
print("=== 创建环境 ===")

# 创建单个环境
env = gym.make('CartPole-v1')
print(f"环境名称: CartPole-v1")
print(f"观察空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

# 创建向量环境（用于并行训练）
vec_env = make_vec_env('CartPole-v1', n_envs=4)
print(f"\n向量环境创建完成，环境数量: {vec_env.num_envs}")

# 创建PPO模型
print("\n=== 创建PPO模型 ===")

# PPO参数说明:
# policy: 策略网络类型（MlpPolicy, CnnPolicy等）
# env: 环境
# learning_rate: 学习率
# n_steps: 每轮收集的步数
# batch_size: 批次大小
# gamma: 折扣因子
# gae_lambda: GAE lambda参数
# ent_coef: 熵系数（鼓励探索）
# verbose: 日志详细程度

model = PPO(
    policy='MlpPolicy',
    env=vec_env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log="./ppo_cartpole_tensorboard/"
)

print("\nPPO模型结构:")
print(model.policy)

# 训练模型
print("\n=== 训练模型 ===")

# 训练参数
total_timesteps = 100000

print(f"开始训练，总步数: {total_timesteps}")

# 训练模型
model.learn(total_timesteps=total_timesteps)

print("\n训练完成！")

# 保存模型
print("\n=== 保存模型 ===")

model.save("ppo_cartpole")
print("模型已保存为 ppo_cartpole.zip")

# 加载模型
loaded_model = PPO.load("ppo_cartpole")
print("模型已加载")

# 评估模型
print("\n=== 评估模型 ===")

# 创建评估环境
eval_env = Monitor(gym.make('CartPole-v1'))

# 评估模型
mean_reward, std_reward = evaluate_policy(loaded_model, eval_env, n_eval_episodes=10)

print(f"平均奖励: {mean_reward:.2f}")
print(f"奖励标准差: {std_reward:.2f}")

# 可视化训练过程
print("\n=== 可视化训练过程 ===")

# 运行一个完整的episode并收集数据
obs = eval_env.reset()
done = False
total_reward = 0
observations = []

while not done:
    action, _states = loaded_model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    total_reward += reward
    observations.append(obs)
    done = terminated or truncated

print(f"单Episode奖励: {total_reward}")

# 可视化观察数据
observations = np.array(observations)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 小车位置
axes[0, 0].plot(observations[:, 0])
axes[0, 0].set_title('小车位置')
axes[0, 0].grid(True, alpha=0.3)

# 小车速度
axes[0, 1].plot(observations[:, 1])
axes[0, 1].set_title('小车速度')
axes[0, 1].grid(True, alpha=0.3)

# 杆角度
axes[1, 0].plot(observations[:, 2])
axes[1, 0].set_title('杆角度')
axes[1, 0].grid(True, alpha=0.3)

# 杆角速度
axes[1, 1].plot(observations[:, 3])
axes[1, 1].set_title('杆角速度')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 自定义策略网络
print("\n=== 自定义策略网络 ===")

from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """自定义CNN特征提取器"""

    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space, features_dim)
        
        # 假设输入是图片（84x84x4）
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # 计算特征维度
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]
        
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations):
        return self.linear(self.cnn(observations))

# 使用自定义策略
print("\n创建使用自定义策略的PPO模型...")

# 注意：自定义策略需要图像输入环境，这里只是示例
# custom_model = PPO(
#     policy='CnnPolicy',
#     env=image_env,
#     policy_kwargs=dict(features_extractor_class=CustomCNN),
#     verbose=1
# )

# 超参数调优示例
print("\n=== 超参数调优 ===")

# 定义不同的超参数组合
param_combinations = [
    {'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64},
    {'learning_rate': 1e-4, 'n_steps': 1024, 'batch_size': 32},
    {'learning_rate': 5e-4, 'n_steps': 4096, 'batch_size': 128},
]

results = []

for params in param_combinations:
    print(f"\n训练参数: {params}")
    
    # 创建模型
    trial_model = PPO(
        policy='MlpPolicy',
        env=vec_env,
        **params,
        verbose=0
    )
    
    # 训练
    trial_model.learn(total_timesteps=50000)
    
    # 评估
    mean_reward, _ = evaluate_policy(trial_model, eval_env, n_eval_episodes=5)
    results.append({
        'params': params,
        'mean_reward': mean_reward
    })
    
    print(f"平均奖励: {mean_reward:.2f}")

# 对比结果
print("\n超参数调优结果:")
for result in sorted(results, key=lambda x: x['mean_reward'], reverse=True):
    print(f"参数: {result['params']} -> 奖励: {result['mean_reward']:.2f}")

# 训练多个环境
print("\n=== 多环境训练 ===")

# 创建多个环境
multi_env = make_vec_env(['CartPole-v1', 'Pendulum-v1'], n_envs=2)

# 注意：不同环境需要相同的观察和动作空间
# 这里只是演示多环境的创建方式

print(f"多环境创建完成: {multi_env.num_envs}个环境")

# 清理
eval_env.close()
vec_env.close()

# 删除保存的模型文件
import os
for f in ['ppo_cartpole.zip', 'ppo_cartpole.pth']:
    if os.path.exists(f):
        os.remove(f)
        print(f"已删除: {f}")

print("\nPPO算法学习完成！")