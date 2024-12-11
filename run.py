import time

import gymnasium as gym
import numpy as np
import gymnasium_envs
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
import yaml
import torch as th

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
    normalization_range=config['normalization_range'],
)
model_path = './logs/checkpoint/Reach-v0-PPO-20241211103637/Reach-v0-PPO-20241211103637_2000000_steps.zip'
if config['alg'] == 'PPO':
    model = PPO.load(model_path, env=env)
elif config['alg'] == 'TD3':
    model = TD3.load(model_path, env=env)
obs, _ = env.reset()
i = 0
while i < 10000:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    if done:
        env.reset()
    # obs_record = np.r_[obs_record, [obs]]
    i += 1
    # print(i)

# t = 0
# curr_time = time.time()
# while True:
#     obs, reward, terminated, _, info = env.step(np.zeros(240))
#     t += 1
#     print(t)
#     if t == 2010:
#         env.reset()
#         print('one loop')
#         print('time cost:', time.time() - curr_time)
#         t = 0
