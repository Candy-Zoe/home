# Gymnasium自定义环境学习
# 主要内容：创建自定义环境、实现核心方法

import gymnasium as gym
from gymnasium import spaces
import numpy as np

print("=== 创建自定义环境 ===")

class SimpleEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.state = None
    
    def _get_obs(self):
        return self.state
    
    def _get_info(self):
        return {'distance': np.linalg.norm(self.state)}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.random.uniform(-1, 1, size=(2,))
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        if action == 0:
            self.state[0] += 0.1
        elif action == 1:
            self.state[0] -= 0.1
        elif action == 2:
            self.state[1] += 0.1
        
        reward = -np.linalg.norm(self.state)
        terminated = np.linalg.norm(self.state) < 0.1
        truncated = False
        
        if self.render_mode == 'human':
            self.render()
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        if self.render_mode == 'human':
            print(f"当前状态: {self.state}")

print("\n=== 使用自定义环境 ===")
env = SimpleEnv()
observation, info = env.reset()
print(f"初始状态: {observation}")

total_reward = 0
for _ in range(10):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    print(f"动作: {action}, 状态: {observation}, 奖励: {reward}")
    
    if terminated:
        print("达到目标！")
        break

print(f"总奖励: {total_reward}")
env.close()

print("\n=== 包装器示例 ===")
wrapped_env = gym.wrappers.TimeLimit(SimpleEnv(), max_episode_steps=5)
observation, info = wrapped_env.reset()
for _ in range(10):
    observation, reward, terminated, truncated, info = wrapped_env.step(wrapped_env.action_space.sample())
    if terminated or truncated:
        print(f"结束原因: terminated={terminated}, truncated={truncated}")
        break