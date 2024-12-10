import time

import gymnasium as gym
import numpy as np
import gymnasium_envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CallbackList, BaseCallback, CheckpointCallback, EvalCallback
import yaml
import torch as th

with open('config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)

currenttime = int(time.time())
currenttime = time.strftime("%Y%m%d%H%M%S", time.localtime(currenttime))


env = gym.make(
    config['task_name'],
    render=config['render'],
    xml_path=config['xml_path'],
    xml_file_name=config['xml_file_name'],
    basic_skills=config['basic_skill_name'],
    specified_skills=config['specified_skill_name'],
    kitchen_tasks_name=config['kitchen_tasks_name'],
    kitchen_tasks_chain=config['kitchen_tasks_chain'],
    normalization_range=config['normalization_range'],
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=config['log_path'] + 'eval/' + config['task_name'] + '-' + config['alg'] + '-' + currenttime + '/',
    log_path=config['log_path'] + 'tensorboards/' + config['task_name'] + '-' + config['alg'] + '-' + currenttime + '/',
    eval_freq=20000,
    deterministic=True,
    render=False,
)
print('saveing best model in training.')
checkpoint_callback = CheckpointCallback(
                save_freq=200000,
                save_path=config['log_path'] + 'checkpoint/' + config['task_name'] + '-' + config['alg'] + '-' + currenttime + '/',
                name_prefix=config['task_name'] + '-' + config['alg'] + '-' + currenttime,
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
print('saveing checkpoint model in training.')
callback = CallbackList([eval_callback, checkpoint_callback])

logs_file = config['log_path'] + 'tensorboards/' + config['task_name'] + '-' + config['alg'] + '-' + currenttime
check_env(env)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=[256, 256], vf=[256, 256]))
if config['alg'] == 'PPO':
    print('PPO algorithm training.')
    model = PPO(
        policy_kwargs=policy_kwargs,
        policy='MultiInputPolicy',
        env=env,
        verbose=1,
        learning_rate=0.0003,
        ent_coef=0.0016,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        tensorboard_log=logs_file,
    )
    model.learn(
        total_timesteps=3000000,
        tb_log_name=config['alg'] + '-' + currenttime,
        callback=callback,
    )
    print('training done')
    model.save(config['model_path'] + config['alg'] + '/' + config['task_name'] + '-' + currenttime + '.pkl')

#
# env.reset()
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
