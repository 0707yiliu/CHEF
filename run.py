import time

import gymnasium as gym
import numpy as np

import gymnasium_envs
from stable_baselines3 import PPO

env = gym.make("Chef-v0")
env.reset()
while True:
    obs, reward, terminated, _, info = env.step(np.zeros(14))