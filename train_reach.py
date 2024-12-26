import logging
import time

import gymnasium as gym
import torch as th
import yaml
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env

from gymnasium_envs.utils import linear_schedule

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
from stable_baselines3.common.monitor import Monitor
env = Monitor(env)

eval_callback = EvalCallback(
    env,
    best_model_save_path=config['alg']['log_path'] + 'eval/' + config['task_name'] + '-' + config['alg']['name'] + '-' + currenttime + '/',
    log_path=config['alg']['log_path'] + 'tensorboards/' + config['task_name'] + '-' + config['alg']['name'] + '-' + currenttime + '/',
    eval_freq=config['alg']['eval_freq'],
    deterministic=True,
    render=False,
)
print('saveing best model in training.')
checkpoint_callback = CheckpointCallback(
                save_freq=config['alg']['checkpoint_freq'],
                save_path=config['alg']['log_path'] + 'checkpoint/' + config['task_name'] + '-' + config['alg']['name'] + '-' + currenttime + '/',
                name_prefix=config['task_name'] + '-' + config['alg']['name'] + '-' + currenttime,
                save_replay_buffer=True,
                save_vecnormalize=True,
            )
print('saveing checkpoint model in training.')
callback = CallbackList([eval_callback, checkpoint_callback])

logs_file = config['alg']['log_path'] + 'tensorboards/' + config['task_name'] + '-' + config['alg']['name'] + '-' + currenttime
check_env(env)
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=dict(pi=config['alg']['latent_networks'], vf=config['alg']['latent_networks']))
if config['alg']['name'] == 'TD3':
    print('PPO algorithm training.')
    model = TD3(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        learning_rate=0.0003,
        batch_size=64,
        tensorboard_log=logs_file,
    )
elif config['alg']['name'] == 'PPO':
    model = PPO(
        policy_kwargs=policy_kwargs,
        policy=config['alg']['policy'],
        env=env,
        verbose=1,
        # target_kl=config['alg']['target_kl'],
        clip_range=linear_schedule(initial_value=config['alg']['clip_range'][0], lowest_value=config['alg']['clip_range'][1], up=True),
        learning_rate=linear_schedule(initial_value=config['alg']['learning_rate']),
        ent_coef=config['alg']['ent_coef'],
        n_steps=config['alg']['n_steps'],
        batch_size=config['alg']['batch_size'],
        n_epochs=config['alg']['n_epochs'],
        gamma=config['alg']['gamma'],
        tensorboard_log=logs_file,
    )

logger = logging.getLogger('alg_logger')
logger.setLevel(logging.DEBUG)
alg_config_file_handler = logging.FileHandler(config['alg']['log_path'] + 'alg_config/' + config['task_name'] + '-' + config['alg']['name'] + '-' + currenttime + '.log')
alg_config_file_handler.setLevel(logging.DEBUG)
logger.addHandler(alg_config_file_handler)
logger.info(f'training algorithm:' + str(config['alg']['name']))
logger.info(f'latent network:' + str(config['alg']['latent_networks']))
logger.info(f'target KL:' + str(config['alg']['target_kl']))
logger.info(f'entropy coefficient:' + str(config['alg']['ent_coef']))
logger.info(f'number of step each epoch:' + str(config['alg']['n_steps']))
logger.info(f'batch size each episode:' + str(config['alg']['batch_size']))
logger.info(f'number of epoch:' + str(config['alg']['n_epochs']))
logger.info(f'clip range:' + str(config['alg']['clip_range']))
logger.info(f'learning rate:' + str(config['alg']['learning_rate']))
logger.info(f'reward gamma:' + str(config['alg']['gamma']))
logger.info(f'robot EEF range - ' +
            'ee_pos_limitation_low: ' + str(config['robot']['ee_pos_limitation_low']) +
            'ee_pos_limitation_high: ' + str(config['robot']['ee_pos_limitation_high']) +
            'ee_rot_limitation_low: ' + str(config['robot']['ee_rot_limitation_low']) +
            'ee_rot_limitation_high: ' + str(config['robot']['ee_rot_limitation_high']) +
            'ee_pos_increment: ' + str(config['robot']['ee_pos_increment']) +
            'ee_rot_increment: ' + str(config['robot']['ee_rot_increment'])
            )
logger.info(f'goal range - ' +
            'goal_max_pos: ' + str(config['task']['goal_max_pos']) +
            'goal_min_pos: ' + str(config['task']['goal_min_pos']) +
            'done_limit: ' +  str(config['task']['done_limit'])
            )

model.learn(
    total_timesteps=config['alg']['total_timesteps'],
    tb_log_name=config['alg']['name'] + '-' + currenttime,
    callback=callback,
)
print('training done')
model.save(config['alg']['model_path'] + config['alg']['name'] + '/' + config['task_name'] + '-' + currenttime + '.pkl')

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
