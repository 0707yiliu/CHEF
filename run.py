import time

import gymnasium as gym
import numpy as np
import gymnasium_envs
from stable_baselines3 import PPO
import yaml

with open('config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)


env = gym.make(
    config['task_name'],
    render=True,
    xml_path=config['xml_path'],
    xml_file_name=config['xml_file_name'],
    basic_skills=config['basic_skill_name'],
    specified_skills=config['specified_skill_name'],
    kitchen_tasks_name=config['kitchen_tasks_name'],
    kitchen_tasks_chain=config['kitchen_tasks_chain'],
)
env.reset()
t = 0
while True:
    obs, reward, terminated, _, info = env.step(np.zeros(14))
    t += 1
    if t == 1500:
        env.reset()
        t = 0
