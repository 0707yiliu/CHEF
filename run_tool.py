import time

import gymnasium as gym
import numpy as np
import gymnasium_envs
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
import yaml
import torch as th

with open('config/chef_v1.yml', 'r', encoding='utf-8') as cfg:
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
log_path = config['alg']['log_path']
# model_path = log_path + 'eval/Chef-v1-PPO-20250121105615/best_model.zip'
# date Chef-v1-20250124093548.pkl is one skill learning
model_path = '/home/yi/robotic_manipulation/CHEF/models/PPO/Chef-v1-20250127013906.pkl'

# model_path = './models/PPO/Chef-v0-20241219183544.pkl'
if config['alg']['name'] == 'PPO':
    model = PPO.load(model_path, env=env)
elif config['alg']['name'] == 'TD3':
    model = TD3.load(model_path, env=env)
obs, _ = env.reset()
i = 0
reset_flag = False
obs_buffer = np.empty_like(obs)

import matplotlib.pyplot as plt
def plot_data(data):
    fig = plt.figure(figsize=(10, 10)) # robot plot
    for plt_index in range(1, 8):
        ax = fig.add_subplot(3, 3, plt_index)

        ax.plot(data[:, plt_index-1])  # robot plot

        ax.plot(data[:, plt_index + 7 - 1])  # dmp plot

    plt.show()

while True:
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    obs_buffer = np.vstack([obs_buffer, obs])

    # print(action)
    if done:
        reset_flag = True
    # obs_record = np.r_[obs_record, [obs]]
    i += 1
    # print(i)
    if i > config['max_step_one_episode']:
        i = 0
        reset_flag = True
    if reset_flag is True:
        env.reset()
        # plot_data(obs_buffer)
        obs_buffer = np.empty_like(obs)
        reset_flag = False

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
