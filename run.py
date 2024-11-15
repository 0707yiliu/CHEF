import gymnasium as gym
import gymnasium_envs
from stable_baselines3 import PPO

env = gym.make("Chef-v0", render_mode="human")
print(env.observation_space)