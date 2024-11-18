import gymnasium as gym
import numpy as np

import gymnasium_envs
from stable_baselines3 import PPO

env = gym.make("Chef-v0")
env.reset()
while 1:
    env.step(np.zeros(14))