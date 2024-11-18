from typing import Dict, Any, Union

import numpy as np

from gymnasium_envs.envs.core import Task

class KitchenMultiTask(Task):
    def __init__(self,
                 sim) -> None:
        super().__init__(sim)
        self.goal = self._sample_goal()

    def _sample_goal(self) -> np.ndarray:
        """TODO: the goal need to be defined by the task (one goal state + current demonstration state)"""
        goal = np.random.uniform(-1, 1)
        return goal

    def compute_reward(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        pass

    def get_achieved_goal(self) -> np.ndarray:
        return np.array(self.get_site_position('obj'))

    def get_obs(self) -> np.ndarray:
        return np.array([])

    def is_success(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        pass

    def reset(self) -> None:
        pass

    def get_desired_goal(self):
        pass
