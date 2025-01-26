from typing import Dict, Any, Union

import numpy as np

from gymnasium_envs.envs.core import Task

from gymnasium_envs.utils import circle_sample, _normalization, euclidean_distance, cosine_distance, euler_angle_distance
from scipy.spatial.transform import Rotation
import yaml

class KitchenMultiTask(Task):
    with open('config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 switch_gate: bool = True,
                 basic_skills: list = [],
                 specified_skills: list = [],
                 kitchen_tasks_name: list = [],
                 kitchen_tasks_chain: dict = {},
                 normalization_range: list = [0, 1],
                 ) -> None:
        super().__init__(sim)
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
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

        # self.reset_goal_max_pos = [-0.5 + 0.4, -0.1, 1.15]  # hard code the reset goal, related to the circle_sample func
        # self.reset_goal_min_pos = [-0.5 - 0.4, -0.6, 1]
        self.reset_goal_max_pos = self.config['task'][
            'goal_max_pos']  # hard code the reset goal, related to the circle_sample func
        self.reset_goal_min_pos = self.config['task']['goal_min_pos']
        self.basic_robot = np.zeros(3)
        #  the goal of the target task
        self.goal = self._sample_goal()


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

        # success count
        self.success_done_hold_max = 3
        self.success_item = 1
        self._success_done_count = 10
        self._success_count = 0
        self.current_count = 0  # count the success number in current loop
        self.reach_done_limit = self.config['task']['done_limit']
        self.reach_done_go = 0.02  # start from done_go, to done_limit as the lowest done.

    def _sample_goal(self) -> np.ndarray:
        """TODO: the goal need to be defined by the task/skill (one goal state + current demonstration state)"""

        # goal = circle_sample(-0.5, 0, 0.5, 0.55, 1.05, 1.1)
        goal = np.random.uniform(self.reset_goal_min_pos + np.array([-0.5, 0, 0.816]), self.reset_goal_max_pos + np.array([-0.5, 0, 0.816]))
        return goal

    def compute_reward(self) -> Union[np.ndarray, float]:
        """
        compute the reward for environment, distance for example
        For different task/skill, the unified reward type would be used:
            Trajectory's distance + Goal distance
            (because we do not know the goal distance for different skill, we get the input directly rather function)
        Returns:
            Float type distance for goal task
        """
        if self.curr_skill is self.specified_skills[2]:  # pour skill, calculate the distance between cube and the area
            cube_pos = self.sim.get_body_position('pourcube')
            rew = euclidean_distance(self.goal, cube_pos)
        else:  # flip and reach skill, calculate the distance between EEF site and target goal, contain rotation
            ee_site_pos = self.sim.get_site_position('attachment_siteL')
            pos_dis = euclidean_distance(self.goal, ee_site_pos)
            # TODO: add rotation, for reach and flip has only one desired rotation posture
            #  flip make x y zero, z 180, reach make x y zero, z 90
            rew = pos_dis

        return rew

    def get_achieved_goal(self) -> np.ndarray:
        """
        For the dual arm case, the achieved goal means the EEF pos of bi-manual hand
        For the single arm case, the achieved goal means the EEF pos of hand
        Returns: the position and orientation of two hands

        """
        if self.curr_skill is self.specified_skills[2]:
            obj_pos = self.sim.get_body_position('pourcube')
        else:
            # obj_pos = self.sim.get_site_position('attachment_siteL')
            obj_pos = self.sim.get_site_position('EEFee_pos')

        # site_quat = self.sim.get_site_quaternion('attachment_siteL')
        # return np.concatenate([site_pos, site_quat]) # single arm
        return obj_pos  # single arm

    def get_obs(self) -> np.ndarray:
        # table_base_z = self.sim.get_body_position(self.table_base_handle)[-1]  # just need the height of the table
        # grasped_object_pos = self.sim.get_body_position(self.target_grasped_obj_handle)
        # grasped_object_qua = self.sim.get_body_quaternion(self.target_grasped_obj_handle)
        # bottle_pos = self.sim.get_body_position(self.bottle_handle)
        # bottle_qua = self.sim.get_body_quaternion(self.bottle_handle)
        # bowl_center_pos = self.sim.get_body_position(self.bowl_center_handle)
        # bowl_side_pos = self.sim.get_site_position(self.bowl_side_handle)
        # fixed_area_pos = self.sim.get_body_position(self.fixed_area_handle)

        #  !the skill imitation by RL. Obs space in env contains one target for unified perspectives
        grab_obj = 'grab_obj'
        grab_obj_pos = self.sim.get_body_position(grab_obj)
        grab_obj_pos += np.random.uniform(low=-np.ones(3) * 0.003, high=np.ones(3) * 0.003)  # make noise for obj pos in observation space
        grab_obj_pos = _normalization(grab_obj_pos, self.reset_goal_max_pos, self.reset_goal_min_pos, range_max=self.norm_max, range_min=self.norm_min)
        # norm_grab_obj_pos = np.clip(norm_grab_obj_pos, self.norm_min, self.norm_max)
        grab_obj_quat = self.sim.get_body_quaternion(grab_obj)
        grab_obj_rot = self.sim.get_body_euler(grab_obj)
        grab_obj_rot += np.random.uniform(low=-np.ones(3) * np.deg2rad(3), high=np.ones(3) * np.deg2rad(3)) # make noise for obj rot in observation space
        grab_obj_quat = Rotation.from_euler('xyz', grab_obj_rot, degrees=False).as_quat()
        grab_obj_quat = _normalization(grab_obj_quat, _max=1, _min=-1, range_max=self.norm_max, range_min=self.norm_min)
        grab_obj_rot = _normalization(grab_obj_rot, _max=np.pi, _min=-np.pi, range_max=self.norm_max, range_min=self.norm_min)
        obs = np.concatenate([grab_obj_pos, grab_obj_quat])
        return obs

    def is_success(
            self,
            achieved_goal_pos: np.ndarray,
            desired_goal_pos: np.ndarray,
            info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float, bool]:
        pos_dis = euclidean_distance(achieved_goal_pos, desired_goal_pos)
        if self.curr_skill is self.specified_skills[1]:  # flip skill, the done need rotation
            # obj_euler = self.sim.get_body_euler('grab_obj')
            obj_euler = self.sim.get_site_euler('obj_state')
            # print(obj_euler, self.sim.get_site_euler('EEFee_pos'))
            # target_rot = np.array([90, 0])
            # rot_dis = cosine_distance(obj_euler[:2], target_rot)
            obj_euler_z = Rotation.from_euler('xyz', obj_euler, degrees=False).as_euler('zyx', degrees=False)[0]
            rot_dis = np.cos(obj_euler_z) + 1 # cosine makes min in [-pi, pi] is zero, max is 2
            pos_dis = euclidean_distance(self.goal, self.sim.get_site_position('obj_state'))
            suc_dis = rot_dis + pos_dis
            # print('flip suc dis:', rot_dis, pos_dis, suc_dis, self.goal, self.sim.get_site_position('obj_state'))
            done = True if suc_dis < 0.13 else False

        else:
            # print('reach suc dis:', pos_dis, self.goal, self.sim.get_site_position('obj_state'))
            done = True if pos_dis < self.reach_done_go else False  # pouring and reach skill, the done do not need rotation

            # if done is True:
            #     self._success_count += 1
            #     if self._success_count >= self.success_item:
            #         self.current_count += 1
            #         if self.current_count > self._success_done_count:
            #             self.current_count = 0
            #             self._success_count = 0
            #             self.success_item += 1
            #     else:
            #         done = False
            #     if self.success_item > self.success_done_hold_max:
            #         self.success_item = self.success_done_hold_max

            # print(pos_dis, self.reach_done_go)
            # print(self.sim.get_body_position('grab_obj'), achieved_goal_pos, desired_goal_pos)
            # print('-------------')
            if self.reach_done_go < self.reach_done_limit:
                self.reach_done_go = self.reach_done_limit
            if done is True:
                self.reach_done_go -= 0.001
                print(
                    f'done!, the reach done go line is {self.reach_done_go}, the lower limit is {self.reach_done_limit}.'
                )


        return done

    def reset(self, skill_index):
        """
        reset the environment for training each skill, each skill has their own environment

        """
        # reloading mj xml file
        # chose new env
        self._success_count = 0
        self.basic_robot = self.sim.get_body_position('baseL')  # the the position of basic for single arm
        self.curr_skill = self.specified_skills[skill_index]
        # print(self.curr_skill, self.last_skill)
        if self.curr_skill == self.last_skill:
            # if the curr skill same as the last skill, pass and reset the env, reload the xml otherwise
            pass
        else:
            self.sim.reload_xml('scene_' + self.curr_skill + '.xml')
            self.last_skill = self.curr_skill
        self.sim.reset() # reset first and set goal and state then, goal sample from the circle
        # self.goal = circle_sample(-0.5, 0, 0.5, 0.55, 1.05, 1.1)
        self.goal = np.random.uniform(self.reset_goal_min_pos + self.basic_robot, self.reset_goal_max_pos + self.basic_robot)
        # print(self.goal)
        # hard code for different skills' environment
        if skill_index == 0 or skill_index == 2:
            self.sim.set_mocap_pos(mocap='grab_obj', pos=self.goal)
        # if skill_index == 2: !the pour cube in 2-env would be changed in robot config
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


class KitchenSingleTool(Task):
    with open('config/chef_v1.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 switch_gate: bool = True,
                 basic_skills: list = [],
                 specified_skills: list = [],
                 kitchen_tasks_name: list = [],
                 kitchen_tasks_chain: dict = {},
                 normalization_range: list = [0, 1],
                 ) -> None:
        super().__init__(sim)
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
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

        # self.reset_goal_max_pos = [-0.5 + 0.4, -0.1, 1.15]  # hard code the reset goal, related to the circle_sample func
        # self.reset_goal_min_pos = [-0.5 - 0.4, -0.6, 1]
        self.reset_goal_max_pos = self.config['task'][
            'goal_max_pos']  # hard code the reset goal, related to the circle_sample func
        self.reset_goal_min_pos = self.config['task']['goal_min_pos']
        self.ee_high = np.array(self.config['robot']['ee_pos_limitation_high'])
        self.ee_low = np.array(self.config['robot']['ee_pos_limitation_low'])
        self.basic_robot = np.zeros(3)
        #  the goal of the target task
        self.goal = self._sample_goal()


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

        # success count
        self.success_done_hold_max = 3
        self.success_item = 1
        self._success_done_count = 10
        self._success_count = 0
        self.current_count = 0  # count the success number in current loop
        self.reach_done_limit = self.config['task']['done_limit']
        self.reach_done_go = 0.015  # start from done_go, to done_limit as the lowest done.

    def _sample_goal(self) -> np.ndarray:
        """TODO: the goal need to be defined by the task/skill (one goal state + current demonstration state)"""

        # goal = circle_sample(-0.5, 0, 0.5, 0.55, 1.05, 1.1)
        goal = np.random.uniform(self.reset_goal_min_pos, self.reset_goal_max_pos)
        return goal

    def compute_reward(self) -> Union[np.ndarray, float]:
        """
        compute the reward for environment, distance for example
        For different task/skill, the unified reward type would be used:
            Trajectory's distance + Goal distance
            (because we do not know the goal distance for different skill, we get the input directly rather function)
        Returns:
            Float type distance for goal task
        """
        if self.curr_skill is self.specified_skills[2]:  # pour skill, calculate the distance between cube and the area
            cube_pos = self.sim.get_body_position('pourcube')
            rew = euclidean_distance(self.goal, cube_pos)
        else:  # flip and reach skill, calculate the distance between EEF site and target goal, contain rotation
            ee_site_pos = self.sim.get_site_position('EEFee_pos')
            pos_dis = euclidean_distance(self.goal, ee_site_pos)
            # TODO: add rotation, for reach and flip has only one desired rotation posture
            #  flip make x y zero, z 180, reach make x y zero, z 90
            rew = pos_dis

        return rew

    def get_achieved_goal(self) -> np.ndarray:
        """
        For the dual arm case, the achieved goal means the EEF pos of bi-manual hand
        For the single arm case, the achieved goal means the EEF pos of hand
        Returns: the position and orientation of two hands

        """
        if self.curr_skill is self.specified_skills[2]:
            obj_pos = self.sim.get_body_position('pourcube')
        else:
            # obj_pos = self.sim.get_site_position('attachment_siteL')
            obj_pos = self.sim.get_site_position('EEFee_pos')

        # site_quat = self.sim.get_site_quaternion('attachment_siteL')
        # return np.concatenate([site_pos, site_quat]) # single arm
        return obj_pos  # single arm

    def get_obs(self) -> np.ndarray:
        # table_base_z = self.sim.get_body_position(self.table_base_handle)[-1]  # just need the height of the table
        # grasped_object_pos = self.sim.get_body_position(self.target_grasped_obj_handle)
        # grasped_object_qua = self.sim.get_body_quaternion(self.target_grasped_obj_handle)
        # bottle_pos = self.sim.get_body_position(self.bottle_handle)
        # bottle_qua = self.sim.get_body_quaternion(self.bottle_handle)
        # bowl_center_pos = self.sim.get_body_position(self.bowl_center_handle)
        # bowl_side_pos = self.sim.get_site_position(self.bowl_side_handle)
        # fixed_area_pos = self.sim.get_body_position(self.fixed_area_handle)

        #  !the skill imitation by RL. Obs space in env contains one target for unified perspectives
        grab_obj = 'obj_state'
        grab_obj_pos = self.sim.get_site_position(grab_obj)
        grab_obj_pos += np.random.uniform(low=-np.ones(3) * 0.005, high=np.ones(3) * 0.005)  # make noise for obj pos in observation space
        # grab_obj_pos = _normalization(grab_obj_pos, self.reset_goal_max_pos, self.reset_goal_min_pos, range_max=self.norm_max, range_min=self.norm_min)
        grab_obj_pos = _normalization(grab_obj_pos, self.ee_high, self.ee_low,
                                      range_max=self.norm_max, range_min=self.norm_min)

        # norm_grab_obj_pos = np.clip(norm_grab_obj_pos, self.norm_min, self.norm_max)
        grab_obj_rot = self.sim.get_site_euler(grab_obj)
        grab_obj_rot += np.random.uniform(low=-np.ones(3) * np.deg2rad(10), high=np.ones(3) * np.deg2rad(10)) # make noise for obj rot in observation space
        grab_obj_quat = Rotation.from_euler('xyz', grab_obj_rot, degrees=False).as_quat()
        grab_obj_quat = _normalization(grab_obj_quat, _max=1, _min=-1, range_max=self.norm_max, range_min=self.norm_min)
        # grab_obj_rot = _normalization(grab_obj_rot, _max=np.pi, _min=-np.pi, range_max=self.norm_max, range_min=self.norm_min)

        obs = np.concatenate([grab_obj_pos, grab_obj_quat])
        return obs

    def is_success(
            self,
            achieved_goal_pos: np.ndarray,
            desired_goal_pos: np.ndarray,
            info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float, bool]:
        pos_dis = euclidean_distance(achieved_goal_pos, desired_goal_pos)
        if self.curr_skill is self.specified_skills[1]:  # flip skill, the done need rotation
            obj_euler = self.sim.get_body_euler('grab_obj', 'zyx') # TODO: change to the bottle state, (obj_state but not grab_obj)

            rot_dis = abs(np.cos(obj_euler[0]) - np.cos(90))
            done = True if rot_dis < 0.1 and pos_dis < 0.005 else False
        else:
            done = True if pos_dis < self.reach_done_go else False  # pouring and reach skill, the done do not need rotation

            # if done is True:
            #     self._success_count += 1
            #     if self._success_count >= self.success_item:
            #         self.current_count += 1
            #         if self.current_count > self._success_done_count:
            #             self.current_count = 0
            #             self._success_count = 0
            #             self.success_item += 1
            #     else:
            #         done = False
            #     if self.success_item > self.success_done_hold_max:
            #         self.success_item = self.success_done_hold_max

            # print(pos_dis, self.reach_done_go)
            if self.reach_done_go < self.reach_done_limit:
                self.reach_done_go = self.reach_done_limit
            if done is True:
                self.reach_done_go -= 0.001
                print(
                    f'done!, the reach done go line is {self.reach_done_go}, the lower limit is {self.reach_done_limit}.')


        return done

    def reset(self, skill_index):
        """
        reset the environment for training each skill, each skill has their own environment

        """
        # reloading mj xml file
        # chose new env
        self._success_count = 0
        # self.basic_robot = self.sim.get_body_position('baseL')  # the the position of basic for single arm
        self.curr_skill = self.specified_skills[skill_index]
        # print(self.curr_skill, self.last_skill)
        if self.curr_skill == self.last_skill:
            # if the curr skill same as the last skill, pass and reset the env, reload the xml otherwise
            pass
        else:
            self.sim.reload_xml('scene_tool_' + self.curr_skill + '.xml')
            print('DO NOT RELOAD MJX file!!!')
            self.last_skill = self.curr_skill
        self.sim.reset() # reset first and set goal and state then, goal sample from the circle
        # self.goal = circle_sample(-0.5, 0, 0.5, 0.55, 1.05, 1.1)
        self.goal = np.random.uniform(self.reset_goal_min_pos, self.reset_goal_max_pos)
        # print(self.goal)
        # hard code for different skills' environment
        if skill_index == 0 or skill_index == 2:
            self.sim.set_mocap_pos(mocap='grab_obj', pos=self.goal)
        # if skill_index == 2: !the pour cube in 2-env would be changed in robot config
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