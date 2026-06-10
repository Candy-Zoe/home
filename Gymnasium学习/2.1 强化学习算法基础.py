# Gymnasium强化学习算法基础学习
# 主要内容：Q-learning、SARSA、策略梯度、DQN基础

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

print("=== 创建环境 ===")
env = gym.make("CartPole-v1", render_mode="rgb_array")
observation, info = env.reset()
print(f"观察空间形状: {env.observation_space.shape}")
print(f"动作空间: {env.action_space}")

print("\n=== Q-learning算法 ===")
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
    
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.q_table.shape[1])
        return np.argmax(self.q_table[state, :])
    
    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state, :])
        
        new_value = old_value + self.lr * (reward + self.gamma * next_max * (1 - done) - old_value)
        self.q_table[state, action] = new_value
        
        if done:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

print("Q-learning代理已创建")

print("\n=== SARSA算法 ===")
class SARSAAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = np.zeros((state_size, action_size))
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
    
    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.q_table.shape[1])
        return np.argmax(self.q_table[state, :])
    
    def learn(self, state, action, reward, next_state, next_action, done):
        old_value = self.q_table[state, action]
        next_value = self.q_table[next_state, next_action]
        
        new_value = old_value + self.lr * (reward + self.gamma * next_value * (1 - done) - old_value)
        self.q_table[state, action] = new_value
        
        if done:
            self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)

print("SARSA代理已创建")

print("\n=== 离散化状态 ===")
def discretize_state(state, bins=(3, 3, 6, 6)):
    lower_bounds = [-4.8, -4, -0.418, -4]
    upper_bounds = [4.8, 4, 0.418, 4]
    
    state_disc = []
    for i in range(len(state)):
        state_disc.append(np.digitize(state[i], np.linspace(lower_bounds[i], upper_bounds[i], bins[i] + 1)) - 1)
    
    return tuple(state_disc)

print("\n=== 训练Q-learning ===")
agent = QLearningAgent(3*3*6*6, 2)
episodes = 500
rewards = []

for episode in range(episodes):
    state, _ = env.reset()
    state = discretize_state(state)
    total_reward = 0
    done = False
    
    while not done:
        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = discretize_state(next_state)
        
        agent.learn(state, action, reward, next_state, done)
        
        state = next_state
        total_reward += reward
    
    rewards.append(total_reward)
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1}, Average Reward: {np.mean(rewards[-100:]):.2f}")

print("\n=== 可视化训练结果 ===")
plt.plot(rewards)
plt.title('Q-learning训练曲线')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()

print("\n=== 关闭环境 ===")
env.close()
print("环境已关闭")