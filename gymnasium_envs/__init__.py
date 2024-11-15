from gymnasium.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gymnasium_envs.envs:GridWorldEnv",
)

register(
    id="Chef-v0",
    entry_point="gymnasium_envs.envs:ChefEnv_v0",
)
