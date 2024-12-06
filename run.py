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
    render=False,
    xml_path=config['xml_path'],
    xml_file_name=config['xml_file_name'],
    basic_skills=config['basic_skill_name'],
    specified_skills=config['specified_skill_name'],
    kitchen_tasks_name=config['kitchen_tasks_name'],
    kitchen_tasks_chain=config['kitchen_tasks_chain'],
    normalization_range=config['normalization_range'],
)
env.reset()
t = 0
curr_time = time.time()
while True:
    obs, reward, terminated, _, info = env.step(np.zeros(240))
    t += 1
    print(t)
    if t == 2010:
        env.reset()
        print('one loop')
        print('time cost:', time.time() - curr_time)
        t = 0
