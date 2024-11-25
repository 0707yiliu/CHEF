from typing import Dict, Any, Union

import numpy as np

from gymnasium_envs.envs.core import Task


class KitchenMultiTask(Task):
    def __init__(self,
                 sim,
                 switch_gate: bool = True,
                 basic_skills: list = [],
                 specified_skills: list = [],
                 ) -> None:
        super().__init__(sim)
        #  multi skills switch for basic and specified
        self.switch_gate = switch_gate #  use the switch mode
        self.basic_skills = basic_skills
        self.specified_skills = specified_skills

        #  the goal of the target task
        self.goal = self._sample_goal()

        # Observation in Task (define in mujoco xml file)
        self.table_base_handle = 'obj_table'  # table base Z position
        self.target_grasped_obj_handle = 'rigid_cube'  # grasped obj
        self.bottle_handle = 'ycb_bottle'  # bottle for flipping
        self.bowl_center_handle = 'bowl'  # bowl for get obj and pouring to fixed area
        self.bowl_side_handle = 'bowlside' # the side of the bowl for grasping
        self.fixed_area_handle = 'fixedArea'  # fixed area for check the obj
        self.fixed_ares_side_range = 0.2 * 2  # the length of area's side

    def _sample_goal(self) -> np.ndarray:
        """TODO: the goal need to be defined by the task/skill (one goal state + current demonstration state)"""

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
        bowl_center_pos = self.sim.get_body_position(self.bowl_center_handle)
        bowl_side_pos = self.sim.get_site_position(self.bowl_side_handle)
        fixed_area_pos = self.sim.get_body_position(self.fixed_area_handle)
        return np.array([])

    def is_success(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        pass

    def reset(self) -> None:
        pass

    def get_desired_goal(self):
        pass
