from typing import Dict, Any, Union

import numpy as np

from gymnasium_envs.envs.core import Task

from gymnasium_envs.utils import circle_sample, _normalization, euclidean_distance, cosine_distance
from scipy.spatial.transform import Rotation
import yaml

class Reach(Task):
    with open('config/tasks/reach_v0.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 normalization_range: list = [0, 1],
                 ) -> None:
        super().__init__(sim)
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]

        #  the goal of the target task
        self.goal = self._sample_goal()

        self.reset_goal_max_pos = [0.05, 0.55, 0.2]  # hard code the reset goal, related to the circle_sample func
        self.reset_goal_min_pos = [-0.05, 0.45, 0.1]
        self.basic_robot = np.zeros(3)

    def _sample_goal(self) -> np.ndarray:
        goal = np.random.uniform(self.reset_goal_min_pos + self.basic_robot,
                                 self.reset_goal_max_pos + self.basic_robot)
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
        ee_site_pos = self.sim.get_site_position('attachment_siteL')
        pos_dis = euclidean_distance(self.goal, ee_site_pos)
        rew = -pos_dis
        return rew

    def get_achieved_goal(self) -> np.ndarray:
        """
        For the dual arm case, the achieved goal means the EEF pos of bi-manual hand
        For the single arm case, the achieved goal means the EEF pos of hand
        Returns: the position and orientation of two hands

        """
        ee_pos = self.sim.get_site_position('attachment_siteL')
        # site_quat = self.sim.get_site_quaternion('attachment_siteL')
        # return np.concatenate([site_pos, site_quat]) # single arm
        return ee_pos  # single arm

    def get_obs(self) -> np.ndarray:
        #  !the skill imitation by RL. Obs space in env contains one target for unified perspectives
        grab_obj = 'grab_obj'
        grab_obj_pos = self.sim.get_body_position(grab_obj)
        norm_grab_obj_pos = _normalization(grab_obj_pos, self.reset_goal_max_pos, self.reset_goal_min_pos, range_max=self.norm_max, range_min=self.norm_min)
        norm_grab_obj_pos = np.clip(norm_grab_obj_pos, self.norm_min, self.norm_max)
        grab_obj_quat = self.sim.get_body_quaternion(grab_obj)
        norm_grab_obj_quat = _normalization(grab_obj_quat, _max=1, _min=-1, range_max=self.norm_max, range_min=self.norm_min)
        obs = np.concatenate([norm_grab_obj_pos, norm_grab_obj_quat])
        return obs

    def is_success(
            self,
            achieved_goal_pos: np.ndarray,
            desired_goal_pos: np.ndarray,
            info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float, bool]:
        pos_dis = euclidean_distance(achieved_goal_pos, desired_goal_pos)
        done = True if pos_dis < self.config['done_limit'] else False
        if done is True:
            print('done')
        return done

    def reset(self, skill_index):
        """
        reset the environment for training each skill, each skill has their own environment

        """
        # reloading mj xml file
        self.basic_robot = self.sim.get_body_position('baseL')  # hard code for getting the position of robot base
        self.sim.reset()  # reset first and set goal and state then, goal sample from the circle
        self.goal = self._sample_goal()
        self.sim.set_mocap_pos(mocap='grab_obj', pos=self.goal)  # hard code for setting the position of target object
        return self.goal