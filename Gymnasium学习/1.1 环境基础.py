# Gymnasium强化学习环境基础学习
# 主要内容：环境创建、基本操作、观察空间、动作空间

# 导入Gymnasium库
import gymnasium as gym
import numpy as np

# 创建环境
print("=== 创建强化学习环境 ===")

# 创建CartPole环境（经典的平衡杆问题）
env = gym.make('CartPole-v1')
print(f"环境名称: CartPole-v1")
print(f"最大episode步数: {env.spec.max_episode_steps}")

# 查看环境信息
print(f"\n观察空间: {env.observation_space}")
print(f"动作空间: {env.action_space}")

# 观察空间和动作空间详解
print("\n=== 空间定义 ===")

# CartPole的观察空间是4维连续空间
# [小车位置, 小车速度, 杆角度, 杆角速度]
obs_space = env.observation_space
print(f"观察空间类型: {type(obs_space)}")
print(f"观察空间下限: {obs_space.low}")
print(f"观察空间上限: {obs_space.high}")

# CartPole的动作空间是离散的（2个动作）
# 0: 向左推, 1: 向右推
action_space = env.action_space
print(f"\n动作空间类型: {type(action_space)}")
print(f"动作数量: {action_space.n}")

# 重置环境
print("\n=== 重置环境 ===")

# 重置环境，返回初始观察
observation, info = env.reset(seed=42)
print(f"初始观察: {observation}")
print(f"环境信息: {info}")

# 执行动作
print("\n=== 执行动作 ===")

# 执行随机动作
action = env.action_space.sample()
print(f"随机动作: {action}")

# 执行动作，返回新的观察、奖励、完成标志、额外信息
observation, reward, terminated, truncated, info = env.step(action)

print(f"新观察: {observation}")
print(f"奖励: {reward}")
print(f"终止标志: {terminated}")
print(f"截断标志: {truncated}")
print(f"完成状态: {terminated or truncated}")
print(f"额外信息: {info}")

# 完整episode示例
print("\n=== 完整Episode示例 ===")

# 重置环境
observation, info = env.reset()

total_reward = 0
step_count = 0
max_steps = 500

print(f"开始Episode，最多执行 {max_steps} 步")

for step in range(max_steps):
    # 选择动作（使用随机策略）
    action = env.action_space.sample()

    # 执行动作
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    # 如果episode结束
    if terminated or truncated:
        print(f"Episode结束！")
        print(f"  总步数: {step_count}")
        print(f"  总奖励: {total_reward}")
        print(f"  终止原因: {'成功存活' if truncated else '任务失败'}")
        break

# 渲染环境
print("\n=== 渲染环境 ===")

# 创建可渲染的环境
env_render = gym.make('CartPole-v1', render_mode='human')

# 重置并渲染
observation, info = env_render.reset()

# 执行几个步骤并渲染
for _ in range(100):
    action = env_render.action_space.sample()
    observation, reward, terminated, truncated, info = env_render.step(action)

    if terminated or truncated:
        observation, info = env_render.reset()

env_render.close()
print("渲染完成，环境已关闭")

# 使用Wrapper修改环境
print("\n=== 环境Wrapper ===")

# 创建带有奖励塑形的环境
class RewardShapingWrapper(gym.Wrapper):
    """奖励塑形Wrapper"""

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        # 添加额外的奖励
        # 鼓励杆保持接近垂直
        angle = observation[2]
        angle_reward = -abs(angle) * 0.5

        shaped_reward = reward + angle_reward

        return observation, shaped_reward, terminated, truncated, info

# 使用Wrapper
env_wrapped = gym.make('CartPole-v1')
env_wrapped = RewardShapingWrapper(env_wrapped)

print("奖励塑形Wrapper已应用")

# 测试Wrapper
observation, info = env_wrapped.reset()
for _ in range(10):
    action = env_wrapped.action_space.sample()
    observation, reward, terminated, truncated, info = env_wrapped.step(action)
    print(f"奖励: {reward:.4f}")

    if terminated or truncated:
        break

env_wrapped.close()

# 离散动作空间
print("\n=== 离散动作空间 ===")

# 创建FrozenLake环境（网格世界导航）
env = gym.make('FrozenLake-v1', render_mode=None)
print(f"FrozenLake环境观察空间: {env.observation_space}")
print(f"FrozenLake环境动作空间: {env.action_space}")

# 动作空间说明
print("\n动作空间说明:")
print("  0: 左 (LEFT)")
print("  1: 下 (DOWN)")
print("  2: 右 (RIGHT)")
print("  3: 上 (UP)")

# 重置并执行几个步骤
observation, info = env.reset()
print(f"\n初始位置: {observation}")

for _ in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"执行动作 {action} -> 新位置: {observation}, 奖励: {reward}")

env.close()

# 连续动作空间
print("\n=== 连续动作空间 ===")

# 创建Pendulum环境（摆锤控制）
env = gym.make('Pendulum-v1', render_mode=None)
print(f"Pendulum环境观察空间: {env.observation_space}")
print(f"Pendulum环境动作空间: {env.action_space}")

# 动作空间范围
action = env.action_space.sample()
print(f"随机动作示例: {action}")
print(f"动作范围: [{env.action_space.low[0]}, {env.action_space.high[0]}]")

# 执行几个步骤
observation, info = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"奖励: {reward:.4f}")

env.close()

# 监控和记录
print("\n=== 环境监控 ===")

# 使用Monitor记录训练过程
env = gym.make('CartPole-v1')

# 执行几个完整的episode并记录
for episode in range(3):
    observation, info = env.reset()
    total_reward = 0

    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Episode {episode+1}: 总奖励 = {total_reward}, 步数 = {step+1}")
            break

env.close()

# 环境的关闭和清理
print("\n=== 环境清理 ===")

env = gym.make('CartPole-v1')
print("环境已创建")

env.close()
print("环境已关闭，资源已释放")

print("\n基础学习完成！")