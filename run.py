import time

import gymnasium as gym
import numpy as np
import gymnasium_envs
from stable_baselines3 import PPO
import yaml

with open('config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)


env = gym.make(
    config['task_name']
)
env.reset()
while True:
    obs, reward, terminated, _, info = env.step(np.zeros(14))