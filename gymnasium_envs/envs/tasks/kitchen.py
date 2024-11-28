from typing import Dict, Any, Union

import numpy as np

from gymnasium_envs.envs.core import Task

from gymnasium_envs.utils import circle_sample


class KitchenMultiTask(Task):
    def __init__(self,
                 sim,
                 switch_gate: bool = True,
                 basic_skills: list = [],
                 specified_skills: list = [],
                 kitchen_tasks_name: list = [],
                 kitchen_tasks_chain: dict = {},
                 ) -> None:
        super().__init__(sim)
        #  multi skills switch for basic and specified
        self.switch_gate = switch_gate  # use the switch mode
        self.basic_skills = basic_skills
        self.specified_skills = specified_skills
        self.kitchen_tasks_name = kitchen_tasks_name
        self.kitchen_tasks_chain = kitchen_tasks_chain
        self.target_task = []
        self.target_task_chain = []

        # training skills, switch number
        self.curr_skill = self.specified_skills[0]
        self.last_skill = self.curr_skill

        #  the goal of the target task
        # self.goal = self._sample_goal()

        # Observation in Task (define in mujoco xml file)
        self.table_base_handle = 'obj_table'  # table base Z position
        self.target_grasped_obj_handle = 'rigid_cube'  # grasped obj
        self.bottle_handle = 'ycb_bottle'  # bottle for flipping
        self.bowl_center_handle = 'bowl'  # bowl for get obj and pouring to fixed area
        self.bowl_side_handle = 'bowlside'  # the side of the bowl for grasping
        self.fixed_area_handle = 'fixedArea'  # fixed area for check the obj
        self.fixed_ares_side_range = 0.2 * 2  # the length of area's side
        # reset range for objects
        self.grasped_obj_pos_range = np.array([[0.1, 0.06], [0.6, 0.64], [0.98, 1]])
        self.bottle_pos_range = np.array([[-0.12, -0.08], [0.6, 0.64], [0.98, 1]])
        self.bowl_pos_range = np.array([[-0.42, -0.47], [0.6, 0.64], [0.78, 0.82]])
        self.fixed_area_pos_range = np.array([[-0.93, 0.97], [0.6, 0.64], [0.78, 0.82]])

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

    def reset(self, skill_index):
        """
        reset the environment for training each skill, each skill has their own environment

        """
        # reloading mj xml file
        # chose new env
        self.curr_skill = self.specified_skills[skill_index]
        # print(self.curr_skill, self.last_skill)
        if self.curr_skill == self.last_skill:
            # if the curr skill same as the last skill, pass and reset the env, reload the xml otherwise
            pass
        else:
            self.sim.reload_xml('scene_' + self.curr_skill + '.xml')
            self.last_skill = self.curr_skill
        self.sim.reset() # reset first and set goal and state then, goal sample from the circle
        self.goal = circle_sample(-0.5, 0, 0.5, 0.55, 1.0, 1.1)
        # print(self.goal)
        # hard code for different skills' environment
        if skill_index == 0 or skill_index == 2:
            self.sim.set_mocap_pos(mocap='grab_obj', pos=self.goal)
        # if skill_index == 2:
        #     cube_pos = np.copy(self.goal)
        #     cube_pos[-1] += 0.35
        #     self.sim.set_mocap_pos(mocap='grab_obj', pos=self.goal)
        #     self.sim.set_mocap_pos(mocap='pourcube', pos=cube_pos)

        return self.goal

        # """
        # reset the environment when complete the task or early stop
        #     reset the task chain from dictionary (target task)
        #     reset all objects state based on target task
        #
        # """
        # # generate the target task for one episode (for task)
        # self.target_task = self.kitchen_tasks_name[np.random.randint(0, len(self.kitchen_tasks_chain))]
        # self.target_task_chain = self.kitchen_tasks_chain[self.target_task]
        # # reset objects
        # fixedarea_pos = np.zeros(3)
        # bowl_pos = np.zeros(3)
        # bottle_pos = np.zeros(3)
        # for i in range(3):
        #     fixedarea_pos[i] = np.random.uniform(self.fixed_area_pos_range[i, 0], self.fixed_area_pos_range[i, 1])
        #     bowl_pos[i] = np.random.uniform(self.bowl_pos_range[i, 0], self.bowl_pos_range[i, 1])
        #     bottle_pos[i] = np.random.uniform(self.bottle_pos_range[i, 0], self.bottle_pos_range[i, 1])
        # if self.target_task == 'pour':
        #     print('----------------')

    def get_desired_goal(self):
        pass
