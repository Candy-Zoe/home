# Gymnasium环境基础学习
# 主要内容：环境创建、交互、观察空间、动作空间

import gymnasium as gym

print("=== 创建环境 ===")
env = gym.make('CartPole-v1')
print(f"环境名称: {env.spec.id}")

print("\n=== 观察空间 ===")
print(f"观察空间类型: {type(env.observation_space)}")
print(f"观察空间形状: {env.observation_space.shape}")
print(f"观察空间范围: low={env.observation_space.low}, high={env.observation_space.high}")

print("\n=== 动作空间 ===")
print(f"动作空间类型: {type(env.action_space)}")
print(f"动作空间大小: {env.action_space.n}")

print("\n=== 重置环境 ===")
observation, info = env.reset()
print(f"初始观察: {observation}")
print(f"额外信息: {info}")

print("\n=== 执行动作 ===")
action = env.action_space.sample()
observation, reward, terminated, truncated, info = env.step(action)
print(f"执行动作: {action}")
print(f"新观察: {observation}")
print(f"奖励: {reward}")
print(f"终止: {terminated}")
print(f"截断: {truncated}")

print("\n=== 运行一个完整episode ===")
observation, info = env.reset()
total_reward = 0
steps = 0

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    steps += 1
    
    if terminated or truncated:
        break

print(f"Episode结束，总奖励: {total_reward}，步数: {steps}")

print("\n=== 关闭环境 ===")
env.close()

print("\n=== 其他环境示例 ===")
envs = ['MountainCar-v0', 'Acrobot-v1', 'Pendulum-v1']
for env_name in envs:
    try:
        env = gym.make(env_name)
        print(f"{env_name}: obs_space={env.observation_space.shape}, action_space={env.action_space}")
        env.close()
    except:
        print(f"{env_name}: 加载失败")