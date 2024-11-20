from typing import Dict, Any, Union

import numpy as np

from gymnasium_envs.envs.core import Task


class KitchenMultiTask(Task):
    def __init__(self,
                 sim) -> None:
        super().__init__(sim)
        self.goal = self._sample_goal()
        # Observation in Task (define in mujoco xml file)
        self.table_base_handle = 'obj_table'  # table base Z position
        self.target_grasped_obj_handle = 'rigid_cube'  # grasped obj
        self.bottle_handle = 'ycb_bottle'  # bottle for flipping
        self.bowl_handle = 'bowl'  # bowl for get obj and pouring to fixed area
        self.fixed_area_handle = 'fixedArea'  # fixed area for check the obj

    def _sample_goal(self) -> np.ndarray:
        """TODO: the goal need to be defined by the task (one goal state + current demonstration state)"""
        goal = np.random.uniform(-1, 1)
        return goal

    def compute_reward(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        pass

    def get_achieved_goal(self) -> np.ndarray:
        """
        For the dual arm case, the achieved goal means the EEF pos in bi-manual hand
        Returns: the position and orientation of two hands

        """
        return np.array(self.sim.get_site_position('obj'))

    def get_obs(self) -> np.ndarray:
        table_base_z = self.sim.get_body_position(self.table_base_handle)[-1]  # just need the height of the table
        grasped_object_pos = self.sim.get_body_position(self.target_grasped_obj_handle)
        grasped_object_qua = self.sim.get_body_quaternion(self.target_grasped_obj_handle)
        bottle_pos = self.sim.get_body_position(self.bottle_handle)
        bottle_qua = self.sim.get_body_quaternion(self.bottle_handle)

        return np.array([])

    def is_success(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        pass

    def reset(self) -> None:
        pass

    def get_desired_goal(self):
        pass
