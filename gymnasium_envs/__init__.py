from gymnasium.envs.registration import register
import yaml
import os
abs_path = os.path.abspath(os.path.dirname(__file__))
# abs_path = abs_path[:abs_path.find('SoftBodyChef')+len('SoftBodyChef')]
abs_path = abs_path[:abs_path.find('CHEF')+len('CHEF')]

with open(abs_path + '/config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
    config = yaml.load(cfg, Loader=yaml.FullLoader)

with open(abs_path + '/config/chef_v1.yml', 'r', encoding='utf-8') as cfg:
    chef_v1_config = yaml.load(cfg, Loader=yaml.FullLoader)

register(
    id="GridWorld-v0",
    entry_point="gymnasium_envs.envs:GridWorldEnv",
)

register(
    id="Chef-v0",
    entry_point="gymnasium_envs.envs:ChefEnv_v0",
    max_episode_steps=config['max_step_one_episode'],
)

register(
    id="Chef-v1",
    entry_point="gymnasium_envs.envs:ChefEnv_v1",
    max_episode_steps=chef_v1_config['max_step_one_episode'],
)

register(
    id="Reach-v0",
    entry_point="gymnasium_envs.envs:ReachEnv_v0",
    max_episode_steps=config['max_step_one_episode'],
)
