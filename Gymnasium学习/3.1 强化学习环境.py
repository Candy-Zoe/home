# Gymnasium强化学习环境学习
# 主要内容：环境创建、智能体训练、Q-Learning、策略迭代

# 导入必要的库
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import random

# Gymnasium基础
print("=== Gymnasium基础 ===")
print(f"Gymnasium版本: {gym.__version__}")

# 创建简单环境
print("\n=== 创建简单环境 ===")
env = gym.make('CartPole-v1')
print(f"环境名称: CartPole-v1")
print(f"观测空间: {env.observation_space}")
print(f"观测空间形状: {env.observation_space.shape}")
print(f"动作空间: {env.action_space}")
print(f"动作数量: {env.action_space.n}")
print(f"奖励范围: {env.reward_range}")

# 环境交互示例
print("\n=== 环境交互示例 ===")
observation, info = env.reset(seed=42)
print(f"初始观测: {observation}")
print(f"观测维度: {len(observation)}")

# 随机策略
print("\n随机策略测试:")
total_reward = 0
for step in range(20):
    action = env.action_space.sample()  # 随机动作
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if step < 5:
        print(f"  步骤 {step+1}: 动作={action}, 奖励={reward:.2f}")
    
    if terminated or truncated:
        print(f"  在第{step+1}步结束")
        break

print(f"总奖励: {total_reward:.2f}")
env.close()

# Q-Learning算法
print("\n=== Q-Learning算法 ===")

# 创建环境
env = gym.make('FrozenLake-v1', is_slippery=False)
print(f"环境: FrozenLake-v1 (确定性版本)")

# 初始化Q表
Q = defaultdict(lambda: np.zeros(env.action_space.n))

# 超参数
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 1000
max_steps = 100

# 训练Q-Learning
print(f"开始训练 {num_episodes} 回合...")
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    
    for step in range(max_steps):
        # epsilon-greedy策略
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        # 执行动作
        next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Q-Learning更新
        best_next_action = np.argmax(Q[next_state])
        td_target = reward + discount_factor * Q[next_state][best_next_action]
        td_error = td_target - Q[state][action]
        Q[state][action] += learning_rate * td_error
        
        state = next_state
        total_reward += reward
        
        if terminated or truncated:
            break
    
    episode_rewards.append(total_reward)
    
    # 打印进度
    if (episode + 1) % 200 == 0:
        avg_reward = np.mean(episode_rewards[-100:])
        print(f"  回合 {episode+1}: 平均奖励 = {avg_reward:.3f}")

# 测试训练结果
print("\n测试Q-Learning训练结果:")
state, _ = env.reset()
print(f"初始状态: {state}")
for step in range(max_steps):
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    print(f"  状态 {state} -> 动作 {action} -> 状态 {next_state}, 奖励 {reward}")
    state = next_state
    if terminated or truncated:
        break

# 训练曲线
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('Q-Learning 训练曲线')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 滑动平均
window = 50
moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards, alpha=0.3, label='原始')
plt.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', label=f'滑动平均({window})')
plt.xlabel('回合')
plt.ylabel('总奖励')
plt.title('Q-Learning 训练曲线（带滑动平均）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

env.close()

# 策略迭代
print("\n=== 策略迭代 ===")

# 创建环境
env = gym.make('Taxi-v3')
print(f"环境: Taxi-v3")

# 初始化策略和价值函数
num_states = env.observation_space.n
num_actions = env.action_space.n

policy = np.ones((num_states, num_actions)) / num_actions  # 均匀策略
V = np.zeros(num_states)

# 策略评估
def policy_evaluation(policy, env, V, theta=0.0001, gamma=0.99, max_iterations=1000):
    """策略评估"""
    for i in range(max_iterations):
        delta = 0
        for state in range(num_states):
            v = 0
            for action in range(num_actions):
                for prob, next_state, reward, done in env.P[state][action]:
                    v += policy[state][action] * prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[state]))
            V[state] = v
        
        if delta < theta:
            break
    return V

# 策略改进
def policy_improvement(policy, V, env, gamma=0.99):
    """策略改进"""
    policy_stable = True
    
    for state in range(num_states):
        old_action = np.argmax(policy[state])
        
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + gamma * V[next_state])
        
        best_action = np.argmax(action_values)
        
        if old_action != best_action:
            policy_stable = False
        
        policy[state] = np.zeros(num_actions)
        policy[state][best_action] = 1.0
    
    return policy, policy_stable

# 策略迭代
print("执行策略迭代...")
max_iterations = 100
for i in range(max_iterations):
    V = policy_evaluation(policy, env, V)
    policy, policy_stable = policy_improvement(policy, V, env)
    
    if (i + 1) % 10 == 0:
        print(f"  迭代 {i+1}: 策略稳定 = {policy_stable}")
    
    if policy_stable:
        print(f"  在第 {i+1} 次迭代收敛")
        break

print(f"最终价值函数范围: [{V.min():.2f}, {V.max():.2f}]")
print(f"策略价值均值: {V.mean():.2f}")

env.close()

# 价值迭代
print("\n=== 价值迭代 ===")

env = gym.make('FrozenLake-v1', is_slippery=False)
num_states = env.observation_space.n
num_actions = env.action_space.n

V_vi = np.zeros(num_states)
policy_vi = np.zeros((num_states, num_actions))
gamma = 0.99
theta = 0.0001

# 价值迭代
print("执行价值迭代...")
for i in range(1000):
    delta = 0
    for state in range(num_states):
        v = V_vi[state]
        
        action_values = np.zeros(num_actions)
        for action in range(num_actions):
            for prob, next_state, reward, done in env.P[state][action]:
                action_values[action] += prob * (reward + gamma * V_vi[next_state])
        
        V_vi[state] = np.max(action_values)
        delta = max(delta, abs(v - V_vi[state]))
    
    if (i + 1) % 100 == 0:
        print(f"  迭代 {i+1}, delta = {delta:.6f}")
    
    if delta < theta:
        print(f"  在第 {i+1} 次迭代收敛")
        break

# 提取策略
for state in range(num_states):
    action_values = np.zeros(num_actions)
    for action in range(num_actions):
        for prob, next_state, reward, done in env.P[state][action]:
            action_values[action] += prob * (reward + gamma * V_vi[next_state])
    
    best_action = np.argmax(action_values)
    policy_vi[state] = np.eye(num_actions)[best_action]

print(f"价值函数: {V_vi.round(2)}")

env.close()

# 多种环境展示
print("\n=== 多种强化学习环境 ===")

# 可用的环境列表
env_names = [
    'CartPole-v1',
    'MountainCar-v0',
    'Acrobot-v1',
    'Pendulum-v1',
    'FrozenLake-v1',
    'Taxi-v3',
    'Blackjack-v1',
    'CliffWalking-v0'
]

print("常用Gymnasium环境:")
for name in env_names:
    try:
        env = gym.make(name)
        print(f"  {name}:")
        print(f"    观测空间: {env.observation_space}")
        print(f"    动作空间: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"  {name}: 不可用 ({e})")

# 自定义环境
print("\n=== 自定义环境 ===")

class SimpleGridEnv(gym.Env):
    """简单的网格环境"""
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, grid_size=5):
        super().__init__()
        self.grid_size = grid_size
        
        # 动作空间: 0=上, 1=右, 2=下, 3=左
        self.action_space = gym.spaces.Discrete(4)
        
        # 观测空间: 智能体位置 (x, y)
        self.observation_space = gym.spaces.Box(
            low=0, high=grid_size-1, shape=(2,), dtype=np.int32
        )
        
        # 起点和终点
        self.start = np.array([0, 0])
        self.goal = np.array([grid_size-1, grid_size-1])
        self.state = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start.copy()
        return self.state, {}
    
    def step(self, action):
        # 根据动作更新位置
        if action == 0:  # 上
            self.state[1] = min(self.state[1] + 1, self.grid_size - 1)
        elif action == 1:  # 右
            self.state[0] = min(self.state[0] + 1, self.grid_size - 1)
        elif action == 2:  # 下
            self.state[1] = max(self.state[1] - 1, 0)
        elif action == 3:  # 左
            self.state[0] = max(self.state[0] - 1, 0)
        
        # 计算奖励
        distance = np.linalg.norm(self.goal - self.state)
        reward = -distance  # 距离越近奖励越高
        
        # 检查是否到达终点
        terminated = np.array_equal(self.state, self.goal)
        if terminated:
            reward = 100
        
        return self.state, reward, terminated, False, {}
    
    def render(self):
        """渲染环境"""
        grid = np.zeros((self.grid_size, self.grid_size))
        grid[self.state[1], self.state[0]] = 1  # 智能体位置
        grid[self.goal[1], self.goal[0]] = 2    # 目标位置
        return grid

# 使用自定义环境
custom_env = SimpleGridEnv(grid_size=5)
print(f"自定义环境创建成功")
print(f"  观测空间: {custom_env.observation_space}")
print(f"  动作空间: {custom_env.action_space}")

# 测试自定义环境
print("\n测试自定义环境:")
obs, _ = custom_env.reset(seed=42)
print(f"初始状态: {obs}")

for step in range(10):
    action = custom_env.action_space.sample()
    obs, reward, terminated, truncated, _ = custom_env.step(action)
    print(f"  步骤 {step+1}: 动作={action}, 状态={obs}, 奖励={reward:.2f}")
    
    if terminated:
        print("  到达目标!")
        break

# 渲染可视化
print("\n环境渲染:")
for episode in range(3):
    obs, _ = custom_env.reset()
    trajectory = [obs.copy()]
    
    for step in range(20):
        action = custom_env.action_space.sample()
        obs, _, terminated, _, _ = custom_env.step(action)
        trajectory.append(obs.copy())
        
        if terminated:
            break
    
    # 可视化轨迹
    trajectory = np.array(trajectory)
    plt.figure(figsize=(6, 6))
    plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-o', label='智能体路径')
    plt.plot(0, 0, 'go', markersize=15, label='起点')
    plt.plot(4, 4, 'r*', markersize=20, label='目标')
    plt.xlim(-0.5, 4.5)
    plt.ylim(-0.5, 4.5)
    plt.grid(True)
    plt.legend()
    plt.title(f'第{episode+1}回合轨迹')
    plt.tight_layout()
    plt.show()

custom_env.close()

# 经验回放
print("\n=== 经验回放 ===")
print("经验回放是DQN等算法的关键技术")
print("1. 将经验存储在回放缓冲区")
print("2. 随机采样小批量进行训练")
print("3. 打破样本间的相关性")
print("4. 提高样本利用效率")

# 简单的经验回放缓冲区
from collections import deque
import random

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def __len__(self):
        return len(self.buffer)

# 使用经验回放
buffer = ReplayBuffer(capacity=1000)
print("经验回放缓冲区创建完成")

# 模拟存储一些经验
for _ in range(100):
    state = np.random.randn(4)
    action = random.randint(0, 3)
    reward = random.random()
    next_state = np.random.randn(4)
    done = random.random() < 0.1
    buffer.push(state, action, reward, next_state, done)

print(f"缓冲区大小: {len(buffer)}")
batch = buffer.sample(32)
print(f"采样批次: states shape={batch[0].shape}")

# 总结
print("\n=== Gymnasium强化学习学习总结 ===")
print("1. Gymnasium基础（环境创建、交互）")
print("2. 多种经典RL环境")
print("3. Q-Learning算法")
print("4. 策略迭代算法")
print("5. 价值迭代算法")
print("6. 自定义环境")
print("7. 经验回放技术")
print("8. 训练曲线可视化")
print("9. 智能体评估")
print("10. 环境渲染")

print("\nGymnasium强化学习环境学习完成！")
