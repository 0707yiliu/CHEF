from gymnasium_envs.envs.core import Task

class KitchenMultiTask(Task):
    def __init__(self,
                 sim) -> None:
        super().__init__(sim)
