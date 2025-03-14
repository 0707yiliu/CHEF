import random

import numpy as np
import math
from gymnasium_envs.envs.core import MJRobot
from gymnasium import spaces

from gymnasium_envs.utils import _normalization, interp_preprocessed_data_with_vel, euclidean_distance, lowpass_filter, cosine_distance, reward_rescaling, cus_log, euler_angle_distance, quat_dis
from scipy.spatial.transform import Rotation
from gymnasium_envs.DMPs import dmps
from gymnasium_envs.admittance_controller.core import FT_controller as AdmController
import os
import logging
import yaml
import matplotlib.pyplot as plt

logging.basicConfig(format='%(asctime)s - %(levelname)s: %(message)s',
                    level=logging.DEBUG)

class dualUR5e(MJRobot):
    def __init__(self,
                 sim,
                 control_type: str = 'ee',
                 control_ee_rot: bool = True,
                 dmps_weights_num: int = 10,
                 dmps_weights_act: bool = True,
                 control_finger: bool = True,
                 ) -> None:
        self.sensor_num = 4
        # action space definition
        n_actions = 3 if control_type == 'ee' else 6
        n_actions += 3 if control_ee_rot is True else 0
        n_actions += 2 if control_finger is True else 0
        n_actions *= 2 # dual arm
        n_actions *= dmps_weights_num if dmps_weights_act is True else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype=np.float32)
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.R_eef = 'REEF'


        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_list=["lshoulder_pan_joint", "lshoulder_lift_joint", "lelbow_joint", "lwrist_1_joint", "lwrist_2_joint", "lwrist_3_joint", "lfinger",
                        "rshoulder_pan_joint", "rshoulder_lift_joint", "relbow_joint", "rwrist_1_joint", "rwrist_2_joint", "rwrist_3_joint", "rfinger"],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray) -> None:
        self.sim.step()


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_body_position(self.L_eef))
        L_ee_quat = np.copy(self.sim.get_body_quaternion(self.L_eef))
        R_ee_pos = np.copy(self.sim.get_body_position(self.R_eef))
        R_ee_quat = np.copy(self.sim.get_body_quaternion(self.R_eef))
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        R_FT_sensor = self.sim.get_ft_sensor('Rforce', 'Rtorque')
        obs = np.concatenate([L_ee_pos, L_ee_quat, L_FT_sensor,
                              R_ee_pos, R_ee_quat, R_FT_sensor])

        return obs

    def reset(self) -> None:
        self.sim.reset()


class ReachsingleUR5e(MJRobot):
    with open('config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 control_type: str = 'ee',
                 control_ee_rot: bool = True,
                 dmps_weights_num: int = 10,
                 dmps_weights_act: bool = False,
                 dmps_force_enable: bool = False,
                 control_finger: bool = False,
                 normalization_range: list = [0, 1],
                 ) -> None:
        self.max_step_episode = 6000
        self.sensor_num = 4
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
        # action space definition
        n_actions = 3 if control_type == 'ee' else 6
        n_actions += 3 if control_ee_rot is True else 0
        n_actions += 2 if control_finger is True else 0
        # print(n_actions)
        # input()
        action_space = spaces.Box(self.norm_min, self.norm_max, shape=(n_actions,), dtype=np.float32)
        # action space limitation for robot (set the limitation for quick search in RL)
        self.ee_high = np.array(self.config['robot']['ee_pos_limitation_high'])
        self.ee_low = np.array(self.config['robot']['ee_pos_limitation_low'])
        self.ee_rot_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high'])
        self.ee_rot_low = np.deg2rad(self.config['robot']['ee_rot_limitation_low'])
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.env_index = -1
        self.arm_ee_max_pos = np.array([-0.5 + 1, 1, 0.816 + 1.1])
        self.arm_ee_min_pos = np.array([-0.5 - 1, -1, 0.816 - 1.1])
        self.goal = np.zeros(3)
        # DMPs configuration
        self.dmp_x = 0
        self.dmp_n_bfs = dmps_weights_num
        self.dmp_w = np.zeros(self.dmp_n_bfs * 12) if dmps_force_enable is True else np.zeros(self.dmp_n_bfs * 6)  # hard code for shape the size of dmps weight
        self.dmp_force_enable = dmps_force_enable  # the flag for enable force torque DMPs trajectory, delete or add in function
        self.follow_dmp_step = 0  # the step of following DMPs trajectory
        self.dmp_max_step = 2000  # hard code, the DMPs length
        self.ee_pos_increment_range = np.ones(3) * 0.01  # the maximum action step for EEF's pos, for limit the DMPs' random weight
        self.ee_rot_increment_range = np.ones(3) * np.deg2rad(10)  # the maximum action step for EEF's rot, for limit the DMPs' random weight
        self.ee_force_increment_range = np.ones(6) * 10 if dmps_force_enable is True else []
        self.ee_increment_range = np.concatenate((self.ee_pos_increment_range, self.ee_rot_increment_range, self.ee_force_increment_range))
        # admittance controller configuration, hard code of configuration, using critical damping
        self.adm_controller = AdmController(m=0.5, k=1000, kr=5, dt=0.01)
        self.admittance_params = np.zeros((3, 3))  # contains acc, vel and pos in xyz derictions
        self.admittance_paramsT = np.zeros((3, 3))
        self.truncation_num = 3  # Number of digits to be truncated, used when set_action
        self.last_ft = np.zeros(6)  # the last force torque sensor for guiding admittance controller, used be filtered

        self.truncated_num = 0

        self.reward_item = 0

        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            init_qpos=np.deg2rad([90, -135, 90, -90, -90, 0]),
            # init_qpos=np.deg2rad([0,0,0,0,0,0]),
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            actuator_list=['shoulder_panL', 'shoulder_liftL', 'elbowL', 'wrist_1L', 'wrist_2L', 'wrist_3L', 'fingersL'],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )

        self._last_ik_qpos = self.init_qpos
        # self.last_action_ee_pos = np.zeros(6)  # the last (absolute) ee pos and rot, get from reset auto in env==0


    def set_action(self, action: np.ndarray):
        truncated = False
        # fix the action
        action = action.copy()
        act_len = len(action)
        # print(action)
        curr_ee_pos = self.sim.get_site_position('EEFee_pos')  # get the EEF's pos
        curr_ee_rot = self.sim.get_site_euler('EEFee_pos')  # get the EEF's euler
        curr_ee_FT = self.sim.get_ft_sensor('Lforce', 'Ltorque') if self.dmp_force_enable is True else []
        curr_state = np.concatenate((curr_ee_pos, curr_ee_rot, curr_ee_FT))
        # print(curr_ee_rot, self.sim.get_site_quaternion('EEFee_pos'), np.rad2deg(Rotation.from_quat(self.sim.get_site_quaternion('EEFee_pos')).as_euler('xyz', degrees=False)))
        increment_ee_pos = action[:3] * 0.05
        increment_ee_rot = action[3:] * 0.05 * np.pi

        # the limited action is set_dmps_traj (12-dim), which is used as the desired state for admittance controller
        des_pos = np.around(self.last_action_ee_pos + increment_ee_pos, self.truncation_num)  # hard code the getting pos, rot and force/torque
        des_euler = np.around(self.last_action_ee_rot + increment_ee_rot, self.truncation_num)
        # print('init pos:', increment_ee_pos, self.last_action_ee_pos, des_pos)
        des_pos = np.clip(des_pos, self.ee_low + self.sim.get_body_position('baseL'), self.ee_high + self.sim.get_body_position('baseL'))
        des_euler = np.clip(des_euler,  self.ee_rot_low, self.ee_rot_high)
        self.last_action_ee_pos = des_pos
        self.last_action_ee_rot = des_euler
        # print('calculate pos', self.last_action_ee_pos)
        # print(self.last_action_ee_pos - self.sim.get_body_position('baseL'), self.last_action_ee_rot, increment_ee_pos)
        # print(action)
        # print(des_pos)
        # testing the simulator following by DMPs trajectory
        # des_pos = self.dmp_traj[:3, self.follow_dmp_step]
        # des_euler = self.dmp_traj[3:6, self.follow_dmp_step]
        # --------

        pos_d, rot_d, self.admittance_params, self.admittance_paramsT = self.adm_controller.admittance_control(
                                                                                desired_position=des_pos,
                                                                                desired_rotation=des_euler,
                                                                                FT_data=self.last_ft,
                                                                                params_mat=self.admittance_params,
                                                                                paramsT_mat=self.admittance_paramsT,
                                                                                )

        position_d = np.around(pos_d, self.truncation_num)
        rotation_d = np.around(rot_d, self.truncation_num)

        position_d = np.around(des_pos, self.truncation_num)
        rotation_d = np.around(des_euler, self.truncation_num)

        r_target_quat = Rotation.from_euler('xyz', rotation_d, degrees=False).as_quat()
        curr_qpos = np.array([self.sim.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])  # get the qpos without finger
        # ik_qpos = self.sim.inverse_kinematics_kdl(current_joint=curr_qpos,
        #                                           target_position=position_d - self.sim.get_body_position('baseL'),
        #                                           target_orientation=r_target_quat,)
        # print(position_d - self.sim.get_body_position('baseL'), r_target_quat, rotation_d)


        ik_qpos = self.sim.inverse_kinematics_ikfast(target_position=position_d - self.sim.get_body_position('baseL'),
                                                     target_orientation=r_target_quat,
                                                     q_guess=curr_qpos)

        if ik_qpos is None:
            # print('can not get the qpos from ikfast, keep the raw qpos.')
            ik_qpos = curr_qpos

        # if ik_qpos is None:
        # print(position_d - self.sim.get_body_position('baseL'), r_target_quat)
        # print(ik_qpos)
        # for kdl ----
        # if (abs(ik_qpos) > 6.28).any():
        #     logging.warning(f'get too big ik qpos {ik_qpos}')
        #     ik_qpos = self._last_ik_qpos
        # elif (abs(ik_qpos) > np.pi).any():
        #     pos_index = np.where(abs(ik_qpos) > np.pi)[0]
        #     for i in range(len(pos_index)):
        #         if ik_qpos[pos_index[i]] > 0:
        #             ik_qpos[pos_index[i]] -= np.pi
        #         else:
        #             ik_qpos[pos_index[i]] += np.pi
        # else:
        #     ik_qpos = np.around(ik_qpos, self.truncation_num)
        #     self._last_ik_qpos = ik_qpos
        # -------------
        # print(ik_qpos)
        # print('-----------')
        # print(self.sim.get_site_position('attachment_siteL'))
        # print(self.sim.get_body_position('baseL'), self.sim.get_site_position('attachment_siteL') - self.sim.get_body_position('baseL'))
        # print('current ee pos:', curr_ee_pos)
        # print('desired ee pos:', position_d)
        # print('current qpos', curr_qpos)
        # print('ik qpos:', ik_qpos)
        # print('------------')
        # input()
        # ik_qpos = self.init_qpos
        # init ee pos [-0.366      -0.22391024  1.45015816]
        self.sim.control_joints(self.actuator_list[:-1], ik_qpos)
        self.follow_dmp_step += 1
        if self.follow_dmp_step >= self.dmp_max_step:
            self.follow_dmp_step = self.dmp_max_step-1

        self.truncated_num += 1
        # if self.truncated_num % 1000 == 0:
        #     print(self.sim.get_site_position('attachment_siteL'))
        #     print(self.sim.get_site_euler('attachment_siteL'))
        #     print(position_d, self.sim.get_site_position('attachment_siteL'))
        #     # [-0.1   -0.1    1.466] [-0.9   -0.6    0.816]
        #     print(self.ee_high + self.sim.get_body_position('baseL'), self.ee_low + self.sim.get_body_position('baseL'))
        #     # not limit pos: [-0.3612 - 0.2074  1.4409]
        if self.truncated_num > self.max_step_episode:
            self.truncated_num = 0
            truncated = True
        #     print('truncated in action step')
        self.sim.step()
        # return truncated

    def compute_reward(self):
        """
        compute the reward for environment, distance for example
        For different task/skill, the unified reward type would be used:
            Trajectory's distance + Goal distance
            (because we do not know the goal distance for different skill, we get the input directly rather function)
        Returns:
            Float type distance for goal task
        """
        weights = [1, 0.2]  # the weight for all reward factors, index 0: pos and rot err, index 1: trajectory err
        # calculater pos and rot err
        rew_rot = 0  # some envs do not need rotation reward
        if self.env_index == 2:  # pour skill, calculate the distance between cube and the area
            cube_pos = self.sim.get_body_position('pourcube')
            rew_pos = euclidean_distance(self.goal, cube_pos)
            # rew_rot = 0  # disable the rotation distance
        else:  # flip and reach skill, calculate the distance between EEF site and target goal, contain rotation
            ee_site_pos = self.sim.get_site_position('attachment_siteL')
            pos_dis = euclidean_distance(self.goal, ee_site_pos)
            rew_pos = pos_dis
            # if self.env_index == 0:
            #     rew_rot = 0  # disable the rotation distance
            if self.env_index == 1:
                obj_euler = self.sim.get_body_euler('grab_obj')
                rew_rot = euler_angle_distance(self.target_state[3:5], obj_euler[:2]) # get the reward of rotation, only focus on x-aixs and y-axis

        # calculater trajectory err between current state and demonstration's state
        curr_ee_pos = self.sim.get_site_position('attachment_siteL')  # get the EEF's pos
        curr_ee_rot = self.sim.get_site_euler('attachment_siteL')  # get the EEF's euler
        curr_ee_FT = self.sim.get_ft_sensor('Lforce', 'Ltorque') if self.dmp_force_enable is True else []
        curr_state = np.concatenate((curr_ee_pos, curr_ee_rot, curr_ee_FT))
        rew_traj = euclidean_distance(curr_state[:3], self.dmp_traj[:3, self.follow_dmp_step])
        # print('reward for pos:', rew_traj)
        # print('current pos:', curr_ee_pos, curr_ee_rot)
        # print('dmps traj:', self.dmp_traj[:, self.follow_dmp_step], self.follow_dmp_step)
        # input()

        # rew = -weights[0] * (rew_pos + rew_rot) + \
        #       (-weights[1] * rew_traj)

        rew = -weights[0] * (rew_pos + rew_rot)

        rootee_pos = ee_site_pos - self.sim.get_body_position('baseL')
        eef_quat = self.sim.get_site_quaternion('EEFee_pos')
        self.reward_item += 1
        if self.reward_item > 2000:
            logging.info(f'reward each 2000 act: object posture distance is {rew_pos}, {self.goal} and {ee_site_pos} and {eef_quat}')
            # rootee_pos [ 0.134      -0.22391024  0.63415816]
            self.reward_item = 0
        return rew


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_body_position(self.L_eef))
        norm_L_ee_pos = _normalization(L_ee_pos, self.ee_high + self.sim.get_body_position('baseL'), self.ee_low + self.sim.get_body_position('baseL'), range_max=self.norm_max, range_min=self.norm_min)
        norm_L_ee_pos = np.clip(norm_L_ee_pos, -1, 1)
        L_ee_quat = np.copy(self.sim.get_body_quaternion(self.L_eef))
        norm_L_ee_quat = _normalization(L_ee_quat, _max=1, _min=-1, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of quaternion
        L_ee_vels = np.copy(self.sim.get_body_velocity(self.L_eef))
        norm_L_ee_vels = _normalization(L_ee_vels, _max=1, _min=-1, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of velocity
        norm_L_ee_vels = np.clip(norm_L_ee_vels, -1, 1)
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        norm_L_FT_snesor = _normalization(L_FT_sensor, _max=50, _min=-50, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of FT sensor
        norm_L_FT_snesor = np.clip(norm_L_FT_snesor, -1, 1)
        # obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat, norm_L_ee_vels, norm_L_FT_snesor, self.dmp_w])  # put the DMPs weights to observation space
        obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat])
        return obs

    def reset(self, index, target_goal) -> bool:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        print('reset')
        self.goal = target_goal
        self.env_index = index
        self.sim.set_forward()
        _reset_goal = False
        # hard code for different skill's environment
        local_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
        start_pos = np.zeros(3)
        start_rot = np.zeros(3)
        target_rot = np.zeros(3)
        base = self.sim.get_body_position('baseL')
        if self.env_index == 0:  # reach skill,
            qpos_random = np.deg2rad(np.random.uniform(-np.ones(6) * 60, np.ones(6) * 60))
            qpos_random[0] = np.deg2rad(np.random.uniform(-180, 180))
            qpos_random[1] -= 1.57  # make the arm stand upright on the table
            qpos_random = self.init_qpos  # !!!init the qpos to the fix pos
            self.sim.set_joint_qpos(self.joint_list[:-1], qpos_random)  # set the joint randomly
            self.sim.control_joints(self.actuator_list[:-1], qpos_random)
            # testing in the reach env for dmps
            # start_pos, start_quat = self.sim.forward_kinematics_kdl(qpos_random)  # get the reset pos and rot
            self.last_action_ee_pos, last_action_ee_rot = self.sim.forward_kinematics_ikfast(qpos_random)
            self.last_action_ee_rot = Rotation.from_quat(last_action_ee_rot).as_euler('xyz', degrees=False)
            self.last_action_ee_pos += base
            # start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
            target_rot = np.deg2rad(np.random.uniform([-90, 0, -180], [-90, 0, 180]))
            data_path = local_path + '../../datasets/reach/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)


        elif self.env_index == 1:  # flip skill, use IK to generate one posture, fix the Z-rotation face to the ground
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            sim_euler = np.random.uniform(np.deg2rad([-10, -10, -180]), np.deg2rad([10, 10, 180]))
            sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
            q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
            inv_done = False
            sample_times = 0

            data_path = local_path + '../../datasets/flip/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)

            while inv_done is False:
                if q_inv.max() > np.pi or q_inv.min() < -np.pi:
                    sample_times += 1
                    sim_euler = np.random.uniform(np.deg2rad([-10, -10, -180]), np.deg2rad([10, 10, 180]))
                    sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
                    q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
                    if sample_times > 500:
                        _reset_goal = True
                        # print(target_goal)
                        # print('break')
                        break
                else:
                    inv_done = True
                    # print(q_inv, sim_euler, target_goal - base)
                    self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
                    self.sim.control_joints(self.actuator_list[:-1], q_inv)

                    # testing in the reach env for dmps
                    start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
                    start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
                    target_rot = sim_euler
                    target_rot[0] = 90


        elif self.env_index == 2:  # pouring skill, set the position of the ee upon the round of fixed area
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
            sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
            target_goal[-1] += 0.3
            q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
            inv_done = False
            sample_times = 0

            data_path = local_path + '../../datasets/pour/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)

            while inv_done is False:
                if q_inv.max() > np.pi or q_inv.min() < -np.pi:
                    sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
                    # sim_euler = np.random.uniform(np.deg2rad([-95, 0, 0]), np.deg2rad([-85, 10, 10]))
                    sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
                    q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
                    # print('change')
                    sample_times += 1
                    if sample_times > 500:
                        _reset_goal = True
                        # print(target_goal)
                        # print('break')
                        break
                else:
                    inv_done = True
                    # print(q_inv, sim_euler, target_goal - base)
                    self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
                    self.sim.control_joints(self.actuator_list[:-1], q_inv)
                    self.sim.set_forward()
                    cube_pos = self.sim.get_body_position('bowl')
                    cube_pos[-1] += 0.1
                    self.sim.set_mocap_pos(mocap='pourcube', pos=cube_pos)

                    # testing in the reach env for dmps
                    start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
                    start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
                    target_rot = sim_euler
                    target_rot[0] += 180



        # the general part of DMPs' parameters
        # print('start pos:', start_pos, 'base:', base)
        start_pos = start_pos + base
        target_pos = target_goal  # get the reset target goal for dmp
        start_vels, target_vels = np.zeros(6), np.zeros(6)  # do not use velocity now
        start_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        target_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        self.start_state = np.concatenate((start_pos, start_rot, start_forcetorque))
        # print('start pos calculated:', start_pos)
        # input()
        self.target_state = np.concatenate((target_pos, target_rot, target_forcetorque))
        demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
            data_path)

        if self.dmp_force_enable is True:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot, demo_eeft), axis=0)
        else:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot), axis=0)
        self.DMPs = dmps.dmp_discrete_dyn_weight(n_dmps=demonstration_trajs.shape[0],
                                                 n_bfs=self.dmp_n_bfs,
                                                 dt=1.0 / demonstration_trajs.shape[1])
        dmp_w = self.DMPs.learning(demonstration_trajs, plot=False)
        self.dmp_w = _normalization(dmp_w.flatten(), dmp_w.flatten().max(), dmp_w.flatten().min(), range_max=1, range_min=-1)

        self.dmp_traj, _, _ = self.DMPs.reproduce(dyn_w_gate=False,
                                                  initial=self.start_state,
                                                  goal=self.target_state,
                                                  )  # get output of dmps, do not need the gradient
        # simulator forward
        self.sim.set_forward()


        return _reset_goal


class singleTool(MJRobot):
    with open('config/chef_v1.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 control_type: str = 'ee',
                 control_ee_rot: bool = True,
                 dmps_weights_act: bool = True,
                 dmps_force_enable: bool = False,
                 control_finger: bool = False,
                 normalization_range: list = [0, 1],
                 ) -> None:
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
        # action space definition
        n_actions = 3 if control_type == 'ee' else 6
        n_actions += 3 if control_ee_rot is True else 0
        n_actions += 2 if control_finger is True else 0
        action_space = spaces.Box(self.norm_min, self.norm_max, shape=(n_actions,), dtype=np.float32)
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.tool_site = 'EEFee_pos'
        self.env_index = -1
        self.ee_high = np.array(self.config['robot']['ee_pos_limitation_high'])
        self.ee_low = np.array(self.config['robot']['ee_pos_limitation_low'])
        self.ee_rot_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high'])
        self.ee_rot_low = np.deg2rad(self.config['robot']['ee_rot_limitation_low'])
        self.ee_rot_flip_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high_flip'])
        self.ee_rot_flip_low = np.deg2rad(self.config['robot']['ee_rot_limitaion_low_flip'])
        self.goal = np.zeros(3)
        self._base_pos = np.zeros(3)
        self.max_step_one_episode = self.config['max_step_one_episode']
        # DMPs configuration
        self.dmp_max_step = self.config['demonstration_length'] # the DMPs length, should plus with the ratio for action step
        self.dmp_x = 0
        self.dmp_n_bfs = self.config['DMPs_weights_num']
        self.dmp_w = np.zeros(self.dmp_n_bfs * 12) if dmps_force_enable is True else np.zeros(self.dmp_n_bfs * 6)  # hard code for shape the size of dmps weight
        self.dmp_force_enable = dmps_force_enable  # the flag for enable force torque DMPs trajectory, delete or add in function
        self.follow_dmp_step = 0  # the step of following DMPs trajectory

        # self.ee_pos_increment_range = np.ones(3) * 0.015  # the maximum action step for EEF's pos, for limit the DMPs' random weight
        # self.ee_rot_increment_range = np.ones(3) * np.deg2rad(5)  # the maximum action step for EEF's rot, for limit the DMPs' random weight
        # self.ee_force_increment_range = np.ones(6) * 10 if dmps_force_enable is True else []
        # self.ee_increment_range = np.concatenate((self.ee_pos_increment_range, self.ee_rot_increment_range, self.ee_force_increment_range))
        # admittance controller configuration, hard code of configuration, using critical damping
        self.adm_controller = AdmController(m=0.5, k=1000, kr=5, dt=0.01)
        self.admittance_params = np.zeros((3, 3))  # contains acc, vel and pos in xyz derictions
        self.admittance_paramsT = np.zeros((3, 3))
        self.truncation_num = 4  # Number of digits to be truncated, used when set_action
        self.last_ft = np.zeros(6)  # the last force torque sensor for guiding admittance controller, used be filtered
        self._obs_last_ft = np.zeros(6)  # for using low-pass filter in observation space for F/T sensor

        self.truncated_num = 0   # record the step item for stop the epoch
        self._temporal = 0  # record the step item for DMPs in observation
        self._ik_active = 1  # the flag for observation to get the ik is working or not
        self.reward_item = 0
        self._reward_std = 1  # for r_t/self._reward_std to get the scaling reward
        self._reward_mean = 0 # for r_t - self._reward_mean to get the scaling reward
        self._reward_buffer = []  # record the one epoch reward for get new standard variation reward

        # record buffer, for reward calculate the distance between shape
        self._buffer_traj_size = 200
        self._buffer_traj = np.zeros((3, self._buffer_traj_size))


        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            init_qpos=np.deg2rad([90, -135, 90, -90, -90, 0]),  # hard code for the robot, initial posture
            # init_qpos=np.deg2rad([0, 0, 0, 0, 0, 0]),  # hard code for the robot, initial posture
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            actuator_list=['shoulder_panL', 'shoulder_liftL', 'elbowL', 'wrist_1L', 'wrist_2L', 'wrist_3L', 'fingersL'],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray):
        truncated = False  # for stop the episode when get max step
        # fix the action
        # action = action.copy()
        # act_len = len(action)
        if self.env_index == 2:
            increment_ee_pos = action[:3] * self.config['robot']['ee_pos_increment_pour']
            increment_ee_rot = action[3:] * np.deg2rad(self.config['robot']['ee_rot_increment_pour'])
        else:
            increment_ee_pos = action[:3] * self.config['robot']['ee_pos_increment']
            increment_ee_rot = action[3:] * np.deg2rad(self.config['robot']['ee_rot_increment'])

        # the limited action is set_dmps_traj (12-dim), which is used as the desired state for admittance controller
        des_pos = np.around(self.last_action_ee_pos + increment_ee_pos,
                            self.truncation_num)  # getting pos, rot and force/torque

        des_euler = np.around(self.last_action_ee_rot + increment_ee_rot, self.truncation_num)
        # print('init pos:', increment_ee_pos, self.last_action_ee_pos, des_pos)
        des_pos = np.clip(des_pos, self.ee_low, self.ee_high)
        if self.env_index == 0:
            des_euler = np.clip(des_euler, self.ee_rot_low, self.ee_rot_high)
        elif self.env_index == 1:
            des_euler = np.clip(des_euler, self.ee_rot_flip_low, self.ee_rot_flip_high)
        # des_euler = np.deg2rad([30, 90, -100])
        # print('fixed des_euler in set_action')
        des_quat = Rotation.from_euler('zyx', des_euler, degrees=False).as_quat()
        des_quat = np.roll(des_quat, 1)
        # print(des_pos, des_euler, des_quat)
        self.sim.set_mocap_pos('LEEF', des_pos)
        self.sim.set_mocap_quat('LEEF', des_quat)

        self.last_action_ee_pos = np.copy(des_pos)
        self.last_action_ee_rot = np.copy(des_euler)

        self._temporal += 1
        if self._temporal > self.dmp_max_step - 1:
            self._temporal = self.dmp_max_step - 1
        self.truncated_num += 1
        self.sim.step()

    def compute_reward(self):
        """
        compute the reward for environment, distance for example
        For different task/skill, the unified reward type would be used:
            Trajectory's distance + Goal distance
            (because we do not know the goal distance for different skill, we get the input directly rather function)
        Returns:
            Float type distance for goal task
        """
        weights = self.config['reward_shaping']['weights']  # the weight for all reward factors, index 0: pos and rot err, index 1: trajectory err
        log_base = self.config['reward_shaping']['log_base']  # logarithm configuration (0<log_base<1)
        dis_threshold = self.config['reward_shaping']['dis_threshold']  # if the distance lower than dis_threshold (like 10 cm, 0.1), the reward go to positive
        traj_threshold = self.config['reward_shaping']['max_traj_diff']  # same function as dis_threshold for demonstration trajectory from DMPs
        # normalize the distance in negative part and positive part
        max_dis = self.config['reward_shaping']['max_EEF_distance']
        dis_negative_low = cus_log(x_value=max_dis + (1 - dis_threshold), base_x=log_base)  # specified, need to test if changed the parameters
        dis_negative_high = cus_log(x_value=dis_threshold + (1 - dis_threshold), base_x=log_base)
        dis_positive_low = cus_log(x_value=dis_threshold + (1 - dis_threshold), base_x=log_base)
        dis_positive_high = cus_log(x_value=0 + (1 - dis_threshold), base_x=log_base)

        # calculater pos and rot err
        rew_rot = 0  # some envs do not need rotation reward
        if self.env_index == 2:  # pour skill, calculate the distance between cube and the area
            cube_pos = self.sim.get_body_position('rigid_cube')
            rew_pos = euclidean_distance(self.goal, cube_pos)
            # print('rew dis:', rew_pos, self.goal, cube_pos)
            # rew_rot = 0  # disable the rotation distance
        else:  # flip and reach skill, calculate the distance between EEF site and target goal, contain rotation
            if self.env_index == 0:
                rew_rot = euler_angle_distance(self.target_rot, self.sim.get_site_euler(self.tool_site, rot_type='zyx'))
                # print(rew_rot)
            # print(self.target_rot, self.sim.get_site_euler(self.tool_site), self.sim.get_site_quaternion(self.tool_site))
            ee_site_pos = self.sim.get_site_position(self.tool_site)
            pos_dis = euclidean_distance(self.goal, ee_site_pos)
            # print('pos dis:', pos_dis)
            rew_pos = pos_dis
            # if self.env_index == 0:
            #     rew_rot = 0  # disable the rotation distance
            if self.env_index == 1:
                obj_euler = self.sim.get_body_euler('grab_obj', euler_dire='zyx')
                # print('obj rot:', np.rad2deg(obj_euler))
                # rew_rot = euler_angle_distance(self.target_rot, obj_euler)  # get the reward of rotation, only focus on x-aixs and y-axis
                rew_rot = abs(np.cos(obj_euler[0]) - np.cos(self.target_rot[0])) / 2
        # calculater trajectory err between current state and demonstration's state
        curr_ee_pos = self.sim.get_site_position(self.tool_site)  # get the EEF's pos
        curr_ee_rot = self.sim.get_site_euler(self.tool_site)  # get the EEF's euler
        curr_state = np.concatenate((curr_ee_pos, curr_ee_rot))

        rew_pos_traj = euclidean_distance(curr_state[:3], self.dmp_traj[:3, self._temporal])
        # print(rew_pos_traj)
        self._buffer_traj[:, 0] = curr_state[3:6]
        self._buffer_traj = np.roll(self._buffer_traj, -1, axis=1)
        if self._temporal < self._buffer_traj_size:
            rew_rot_traj = 0
        else:
            rew_rot_traj_x = cosine_distance(self._buffer_traj[0, :],
                                             self.dmp_traj[3, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj_y = cosine_distance(self._buffer_traj[1, :],
                                             self.dmp_traj[4, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj_z = cosine_distance(self._buffer_traj[2, :],
                                             self.dmp_traj[5, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj = (rew_rot_traj_x + rew_rot_traj_y + rew_rot_traj_z) / 3
            # print(rew_rot_traj)
        rew_traj = rew_pos_traj + rew_rot_traj
        # rew_traj = rew_pos_traj
        # print('reward - traj pos and rot:', rew_pos_traj, rew_rot_traj)
        # print('current pos:', curr_ee_pos, curr_ee_rot)
        # print('dmps traj:', self.dmp_traj[:, self.follow_dmp_step], self.follow_dmp_step)
        # input()

        th = 0.02  # distance is 2cm for reach skill
        rew_dis = rew_pos + 0.25 * rew_rot
        # print('reward:', rew_pos, rew_rot)
        # norm method
        # rew_dis = rew_pos + rew_rot
        if rew_dis > dis_threshold:  # negative part
            rew_dis_norm = _normalization(cus_log(rew_dis + (1 - dis_threshold), base_x=log_base),
                                              _min=dis_negative_low, _max=dis_negative_high,
                                              range_min=-1, range_max=0)
        else:
            rew_dis_norm = _normalization(cus_log(rew_dis + (1 - dis_threshold), base_x=log_base),
                                              _min=dis_positive_low, _max=dis_positive_high,
                                              range_min=0, range_max=1)
        # print('rew:', rew, cus_log(rew_dis + (1 - dis_threshold), base_x=log_base))

        # rew = weights[0] * rew_dis_norm + \
        #       weights[1] * _normalization(-rew_traj,
        #                                   _min=-traj_threshold, _max=0,
        #                                   range_min=-1, range_max=1)
        # rew = -weights[0] * rew_dis

        # rew = weights[0] * rew_dis_norm



        # rew = -weights[0] * cus_log(rew_pos + (1 - th * 1.3), 10)

        rew = (-weights[0] * rew_dis) + \
              (-weights[1] * rew_traj)

        # rew = -weights[0] * rew_dis

        # rew -= self._reward_mean

        # rew /= (self._reward_std + 1e-8)
        #
        # self._reward_buffer.append(rew)
        # print(self._reward_mean, self._reward_std, rew)

        # rew += 1 if self.env_index == 0 and rew_pos < 0.02 else 0

        self.reward_item += 1
        if self.reward_item > 2000:  # log the reward
            logging.info(f'reward each 2000 act: object posture distance is {rew_pos} and {rew_rot}, trajectory distance is {rew_traj}')
            self.reward_item = 0
        return rew


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_site_position(self.tool_site))
        # print('LEEPOS:', L_ee_pos)
        # print(L_ee_pos-self._base_pos, self.sim.get_body_position('grab_obj')-self._base_pos)
        # print(self.sim.get_site_euler(self.tool_site))
        # print(euclidean_distance(L_ee_pos, self.sim.get_body_position('grab_obj')))
        # print('------------')
        L_ee_pos = _normalization(L_ee_pos, self.ee_high, self.ee_low, range_max=self.norm_max, range_min=self.norm_min)
        L_ee_quat = np.copy(self.sim.get_site_quaternion(self.tool_site))
        L_ee_rot = np.copy(self.sim.get_site_euler(self.tool_site, rot_type='zyx'))
        # print('robot rot:', np.rad2deg(L_ee_rot))
        L_ee_rot = _normalization(L_ee_rot, _max=self.ee_rot_high, _min=self.ee_rot_low, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of quaternion
        # embedding the current time of DMPs' trajectory to observation space as reference trajectory
        next_reference = self._temporal + 1
        if next_reference == self.dmp_max_step:
            next_reference = self.dmp_max_step - 1
        dmp_pos = self.dmp_traj[0:3, next_reference]
        dmp_pos = _normalization(dmp_pos, self.ee_high, self.ee_low, range_max=self.norm_max, range_min=self.norm_min)
        dmp_rot = self.dmp_traj[3:6, next_reference]
        # print('traj rot:', np.rad2deg(dmp_rot))
        dmp_quat = Rotation.from_euler('zyx', dmp_rot, degrees=False).as_quat()
        dmp_rot = _normalization(dmp_rot, _max=np.pi, _min=-np.pi, range_max=self.norm_max, range_min=self.norm_min)
        _item = next_reference / self.dmp_max_step
        # print(_item)
        # obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat, norm_L_ee_vels, norm_L_FT_snesor, self.dmp_w])  # put the DMPs weights to observation space
        # obs = np.concatenate([L_ee_pos, L_ee_rot, L_FT_sensor, dmp_pos, dmp_rot, [_item, self._ik_active]])
        # print(self.sim.get_site_position(self.tool_site), self.dmp_traj[0:3, next_reference])
        obs = np.concatenate([L_ee_pos, L_ee_quat, dmp_pos, dmp_quat, [_item]])  # F/T sensor change too much in observation space
        # obs = np.concatenate([L_ee_pos, L_ee_quat])
        return obs

    def reset(self, index, target_goal) -> bool:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        logging.info('reset')
        # update reward scaling factor by reward buffer
        if self._reward_buffer:
            # print(self._reward_buffer)
            self._reward_mean, self._reward_std = reward_rescaling(self._reward_buffer)
        self._reward_buffer = []  # reset reward buffer for next epoch
        self._ik_active = 1
        self.truncated_num = 0  # reset the count fot step
        self._temporal = 0  # reset the temporal counter for observation
        self.goal = target_goal
        self.env_index = index
        self.sim.set_forward()
        _reset_goal = False
        # hard code for different skill's environment
        local_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
        start_pos = np.zeros(3)
        start_rot = np.zeros(3)
        target_rot = np.zeros(3)

        if self.env_index == 0:  # reach skill,
            # testing in the reach env for dmps
            # start_pos, start_quat = self.sim.forward_kinematics_kdl(qpos_random)  # get the reset pos and rot
            # get ee pos (start and end)
            init_pos = np.random.uniform([-0.1, -0.7, 0.3], [0.7, 0.1, 0.6])
            self.last_action_ee_pos = init_pos
            self.sim.set_mocap_pos('LEEF', self.last_action_ee_pos)  # reset to the init posture
            start_pos = np.copy(self.last_action_ee_pos)
            # get ee rot (start and end)
            self.last_action_ee_rot = np.deg2rad(np.random.uniform(self.config['robot']['ee_rot_limitation_low'],
                                                                   self.config['robot']['ee_rot_limitation_high']))
            self.sim.set_mocap_quat('LEEF', Rotation.from_euler('zyx', self.last_action_ee_rot).as_quat())
            start_rot = np.copy(self.last_action_ee_rot)
            # print(start_rot)
            # start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
            minus = np.random.uniform(-1, 1)
            if minus <= 0:
                self.target_rot = np.deg2rad(np.random.uniform([-15, -15, -175], [15, 15, -165]))
            else:
                self.target_rot = np.deg2rad(np.random.uniform([-15, -15, -95], [15, 15, -85]))
            # self.target_rot = np.deg2rad(np.random.uniform([-10, -10, -95], [10, 10, -85]))
            data_path = local_path + '../../datasets/reach/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)
            print(data_path)


        elif self.env_index == 1:  # flip skill, use IK to generate one posture, fix the Z-rotation face to the ground
            # get ee pos (start and end)
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            self.last_action_ee_pos = target_goal + ee_noise
            self.sim.set_mocap_pos('LEEF', self.last_action_ee_pos)
            start_pos = np.copy(self.last_action_ee_pos)
            self.sim.set_mocap_pos('virtual_goal', target_goal)
            # get ee rot (start and end)
            self.last_action_ee_rot = np.random.uniform(np.deg2rad([-2, -80, -100]), np.deg2rad([2, 80, -80]))
            self.sim.set_mocap_quat('LEEF', Rotation.from_euler('zyx', self.last_action_ee_rot).as_quat())
            start_rot = np.copy(self.last_action_ee_rot)

            minus = np.random.uniform(-1, 1)
            if minus <= 0:
                self.target_rot = np.deg2rad(np.random.uniform([-175, -80, -100], [-165, 80, -80]))
            else:
                self.target_rot = np.deg2rad(np.random.uniform([165, -80, -100], [175, 80, -80]))

            data_path = local_path + '../../datasets/flip/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)
            print(data_path)

        elif self.env_index == 2:  # pouring skill, set the position of the ee upon the round of fixed area
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            self.last_action_ee_pos = target_goal + ee_noise
            self.last_action_ee_pos[-1] += 0.55
            self.sim.set_mocap_pos('LEEF', self.last_action_ee_pos)
            start_pos = np.copy(self.last_action_ee_pos)

            self.last_action_ee_rot = np.random.uniform(np.deg2rad([-80, -5, -175]), np.deg2rad([80, 5, -165]))
            self.sim.set_mocap_quat('LEEF', Rotation.from_euler('zyx', self.last_action_ee_rot).as_quat())
            start_rot = np.copy(self.last_action_ee_rot)

            self.sim.set_forward()
            # set the cube init pos
            cube_pos = np.copy(self.sim.get_body_position('bowl'))
            # print(cube_pos, self.last_action_ee_pos)
            cube_pos[-1] -= 0.15
            self.sim.set_mocap_pos('pourcube', cube_pos)

            self.sim.set_forward()
                # self.sim.step()

            minus = np.random.uniform(-1, 1)
            if minus <= 0:
                self.target_rot = np.deg2rad(np.random.uniform([-100, -100, -175], [-80, -80, -165]))
            else:
                self.target_rot = np.deg2rad(np.random.uniform([80, 80, -175], [100, 100, -165]))

            data_path = local_path + '../../datasets/pour/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)
            print(data_path)
            self.sim.set_forward()
        #     data_path = local_path + '../../datasets/pour/'
        #     data_names = os.listdir(data_path)
        #     data_path = data_path + random.choice(data_names)
        #
        #     while inv_done is False:
        #         if q_inv.max() > np.pi or q_inv.min() < -np.pi:
        #             sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
        #             # sim_euler = np.random.uniform(np.deg2rad([-95, 0, 0]), np.deg2rad([-85, 10, 10]))
        #             sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
        #             q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
        #             # print('change')
        #             sample_times += 1
        #             if sample_times > 500:
        #                 _reset_goal = True
        #                 # print(target_goal)
        #                 # print('break')
        #                 break
        #         else:
        #             inv_done = True
        #             # print(q_inv, sim_euler, target_goal - base)
        #             self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
        #             self.sim.control_joints(self.actuator_list[:-1], q_inv)
        #             self.sim.set_forward()
        #             cube_pos = self.sim.get_body_position('bowl')
        #             cube_pos[-1] += 0.1
        #             self.sim.set_mocap_pos(mocap='pourcube', pos=cube_pos)
        #
        #             # testing in the reach env for dmps
        #             start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
        #             start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
        #             target_rot = sim_euler
        #             target_rot[0] += 180


        # the general part of DMPs' parameters
        # print('start pos:', start_pos, 'base:', base)
        # start_pos = start_pos + base
        if self.env_index == 0 or self.env_index == 1: # reach, flip
            target_pos = self.goal  # get the reset target goal for dmp
        elif self.env_index == 2: # pour
            target_pos = self.last_action_ee_pos
        start_vels, target_vels = np.zeros(6), np.zeros(6)  # do not use velocity now
        start_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        target_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        self.start_state = np.concatenate((start_pos, start_rot, start_forcetorque))
        # print('start pos calculated:', start_pos)
        # input()
        self.target_state = np.concatenate((target_pos, self.target_rot, target_forcetorque))
        demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
            data_path=data_path,
            ex_length=self.dmp_max_step,
        )
        # ! for euler zyx
        demo_ee_rot = Rotation.from_euler('xyz', demo_ee_rot.T).as_euler('zyx').T
        if self.dmp_force_enable is True:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot, demo_eeft), axis=0)
        else:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot), axis=0)
        self.DMPs = dmps.dmp_discrete_dyn_weight(n_dmps=demonstration_trajs.shape[0],
                                                 n_bfs=self.dmp_n_bfs,
                                                 dt=1.0 / demonstration_trajs.shape[1])
        dmp_w = self.DMPs.learning(demonstration_trajs, plot=False)
        self.dmp_w = _normalization(dmp_w.flatten(), dmp_w.flatten().max(), dmp_w.flatten().min(), range_max=1, range_min=-1)

        self.dmp_traj, _, _ = self.DMPs.reproduce(dyn_w_gate=False,
                                                  initial=self.start_state,
                                                  goal=self.target_state,
                                                  )  # get output of dmps, do not need the gradient
        # draw demo trajectory ------
        # self.sim.modify_scene(self.dmp_traj[:3, :])
        # print('--', target_pos, self.goal, start_pos)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(demonstration_trajs[0, :], demonstration_trajs[1, :], demonstration_trajs[2, :])
        # ax.plot(self.dmp_traj[0, :], self.dmp_traj[1, :], self.dmp_traj[2, :])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        # --------------------------
        # simulator forward
        self.sim.set_forward()
        return _reset_goal


class singleUR5e(MJRobot):
    with open('config/chef_v0.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 control_type: str = 'ee',
                 control_ee_rot: bool = True,
                 dmps_weights_act: bool = True,
                 dmps_force_enable: bool = False,
                 control_finger: bool = False,
                 normalization_range: list = [0, 1],
                 ) -> None:
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
        # action space definition
        n_actions = 3 if control_type == 'ee' else 6
        n_actions += 3 if control_ee_rot is True else 0
        n_actions += 2 if control_finger is True else 0
        action_space = spaces.Box(self.norm_min, self.norm_max, shape=(n_actions,), dtype=np.float32)
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.tool_site = 'EEFee_pos'
        self.env_index = -1
        self.ee_high = np.array(self.config['robot']['ee_pos_limitation_high'])
        self.ee_low = np.array(self.config['robot']['ee_pos_limitation_low'])
        self.ee_rot_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high'])
        self.ee_rot_low = np.deg2rad(self.config['robot']['ee_rot_limitation_low'])
        self.ee_rot_flip_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high_flip'])
        self.ee_rot_flip_low = np.deg2rad(self.config['robot']['ee_rot_limitaion_low_flip'])
        self.goal = np.zeros(3)
        self._base_pos = np.zeros(3)
        self.max_step_one_episode = self.config['max_step_one_episode']
        # DMPs configuration
        self.dmp_max_step = self.config['demonstration_length'] # the DMPs length, should plus with the ratio for action step
        self.dmp_x = 0
        self.dmp_n_bfs = self.config['DMPs_weights_num']
        self.dmp_w = np.zeros(self.dmp_n_bfs * 12) if dmps_force_enable is True else np.zeros(self.dmp_n_bfs * 6)  # hard code for shape the size of dmps weight
        self.dmp_force_enable = dmps_force_enable  # the flag for enable force torque DMPs trajectory, delete or add in function
        self.follow_dmp_step = 0  # the step of following DMPs trajectory

        # self.ee_pos_increment_range = np.ones(3) * 0.015  # the maximum action step for EEF's pos, for limit the DMPs' random weight
        # self.ee_rot_increment_range = np.ones(3) * np.deg2rad(5)  # the maximum action step for EEF's rot, for limit the DMPs' random weight
        # self.ee_force_increment_range = np.ones(6) * 10 if dmps_force_enable is True else []
        # self.ee_increment_range = np.concatenate((self.ee_pos_increment_range, self.ee_rot_increment_range, self.ee_force_increment_range))
        # admittance controller configuration, hard code of configuration, using critical damping
        self.adm_controller = AdmController(m=0.5, k=1000, kr=5, dt=0.01)
        self.admittance_params = np.zeros((3, 3))  # contains acc, vel and pos in xyz derictions
        self.admittance_paramsT = np.zeros((3, 3))
        self.truncation_num = 4  # Number of digits to be truncated, used when set_action
        self.last_ft = np.zeros(6)  # the last force torque sensor for guiding admittance controller, used be filtered
        self._obs_last_ft = np.zeros(6)  # for using low-pass filter in observation space for F/T sensor

        self.truncated_num = 0   # record the step item for stop the epoch
        self._temporal = 0  # record the step item for DMPs in observation
        self._ik_active = 1  # the flag for observation to get the ik is working or not
        self.reward_item = 0
        self._reward_std = 1  # for r_t/self._reward_std to get the scaling reward
        self._reward_mean = 0 # for r_t - self._reward_mean to get the scaling reward
        self._reward_buffer = []  # record the one epoch reward for get new standard variation reward

        # record buffer, for reward calculate the distance between shape
        self._buffer_traj_size = 200
        self._buffer_traj = np.zeros((3, self._buffer_traj_size))


        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            init_qpos=np.deg2rad([90, -135, 90, -90, -90, 0]),  # hard code for the robot, initial posture
            # init_qpos=np.deg2rad([0, 0, 0, 0, 0, 0]),  # hard code for the robot, initial posture
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            actuator_list=['shoulder_panL', 'shoulder_liftL', 'elbowL', 'wrist_1L', 'wrist_2L', 'wrist_3L', 'fingersL'],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray):
        truncated = False  # for stop the episode when get max step
        # fix the action
        # action = action.copy()
        # act_len = len(action)

        increment_ee_pos = action[:3] * self.config['robot']['ee_pos_increment']
        increment_ee_rot = action[3:] * np.deg2rad(self.config['robot']['ee_rot_increment'])
        # -0.3485196, 0.615251, -0.3480298, 0.6155287
        # 0.3541676, 0.6117351, 0.3546546, 0.6120177  # TODO: transfer euler ZYX to XYZ by scipy.Rotation
        # the limited action is set_dmps_traj (12-dim), which is used as the desired state for admittance controller
        des_pos = np.around(self.last_action_ee_pos + increment_ee_pos,
                            self.truncation_num)  # getting pos, rot and force/torque
        des_euler = np.around(self.last_action_ee_rot + increment_ee_rot, self.truncation_num)
        # print('init pos:', increment_ee_pos, self.last_action_ee_pos, des_pos)


        des_pos = np.clip(des_pos, self.ee_low + self.sim.get_body_position('baseL'),
                          self.ee_high + self.sim.get_body_position('baseL'))
        # # !clip for training ------
        if self.env_index == 0:
            des_euler = np.clip(des_euler, self.ee_rot_low, self.ee_rot_high)
        elif self.env_index == 1:
            des_euler = np.clip(des_euler, self.ee_rot_flip_low, self.ee_rot_flip_high)
        # # ! ------------------------


        pos_d, rot_d, self.admittance_params, self.admittance_paramsT = self.adm_controller.admittance_control(
                                                                                desired_position=des_pos,
                                                                                desired_rotation=des_euler,
                                                                                FT_data=self.last_ft,
                                                                                params_mat=self.admittance_params,
                                                                                paramsT_mat=self.admittance_paramsT,
                                                                                )
        position_d = np.around(pos_d, self.truncation_num)
        rotation_d = np.around(rot_d, self.truncation_num) # useless now

        position_d = np.around(des_pos, self.truncation_num)
        rotation_d = np.around(des_euler, self.truncation_num)

        r_target_quat = Rotation.from_euler('zyx', rotation_d, degrees=False).as_quat()
        curr_qpos = np.array(
            [self.sim.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])  # get the qpos without finger
        # ik_qpos = self.sim.inverse_kinematics_kdl(current_joint=curr_qpos,
        #                                           target_position=position_d - self.sim.get_body_position('baseL'),
        #                                           target_orientation=r_target_quat,)
        # print(position_d - self.sim.get_body_position('baseL'), r_target_quat, rotation_d)

        ik_qpos = self.sim.inverse_kinematics_ikfast(target_position=position_d - self.sim.get_body_position('baseL'),
                                                     target_orientation=r_target_quat,
                                                     q_guess=curr_qpos)

        if ik_qpos is None:
            # print('can not get the qpos from ikfast, keep the raw qpos.')
            # print(position_d - self.sim.get_body_position('baseL'))
            # ik_qpos = curr_qpos

            # make tiny random degree for get new achievable position
            end3qpos = curr_qpos[3:] + np.random.uniform(low=np.ones(3) * np.deg2rad(self.config['robot']['ee_rot_increment']),
                                                         high=np.ones(3) * np.deg2rad(self.config['robot']['ee_rot_increment']))
            ik_qpos = np.concatenate((curr_qpos[:3], end3qpos))

            pd, rd = self.sim.forward_kinematics_ikfast(ik_qpos)
            self.last_action_ee_pos = pd + self._base_pos
            self.last_action_ee_rot = Rotation.from_quat(rd).as_euler('zyx', degrees=False)

            # eepos, eequat = self.sim.forward_kinematics_ikfast(ik_qpos)
            # print('ik pos failed')
            # self.last_action_ee_pos = np.copy(eepos)
            # self.last_action_ee_rot = Rotation.from_quat(eequat).as_euler('xyz', degrees=False)

            self._ik_active = -1
        else:
            self._ik_active = 1
            # update last action for next action
            self.last_action_ee_pos = np.copy(position_d)
            self.last_action_ee_rot = np.copy(rotation_d)
        # print(self.last_action_ee_pos, self.last_action_ee_rot, self.ee_low + self.sim.get_body_position('baseL'), self.ee_rot_low)
        # ik_qpos = np.around(ik_qpos, self.truncation_num)
        # array([-3.05432619, -2.35619449, -0.17453293, 0.17453293])
        #
        # ik_qpos = self.init_qpos
        # ik_qpos = self.q_inv
        # print(self.sim.get_site_position('EEFee_pos') - self.sim.get_body_position('baseL'), self.sim.get_site_quaternion('EEFee_pos'), self.sim.forward_kinematics_ikfast(self.init_qpos))
        self.sim.control_joints(self.actuator_list[:-1], ik_qpos)
        self._temporal += 1
        if self._temporal > self.dmp_max_step - 1:
            self._temporal = self.dmp_max_step - 1
        self.truncated_num += 1
        if self.truncated_num > self.max_step_one_episode:  # early stop / get
            truncated = True
            self.truncated_num = 0
            # print('early stop')
        self.sim.step()
        # return truncated

    def compute_reward(self):
        """
        compute the reward for environment, distance for example
        For different task/skill, the unified reward type would be used:
            Trajectory's distance + Goal distance
            (because we do not know the goal distance for different skill, we get the input directly rather function)
        Returns:
            Float type distance for goal task
        """
        weights = self.config['reward_shaping']['weights']  # the weight for all reward factors, index 0: pos and rot err, index 1: trajectory err
        log_base = self.config['reward_shaping']['log_base']  # logarithm configuration (0<log_base<1)
        dis_threshold = self.config['reward_shaping']['dis_threshold']  # if the distance lower than dis_threshold (like 10 cm, 0.1), the reward go to positive
        traj_threshold = self.config['reward_shaping']['max_traj_diff']  # same function as dis_threshold for demonstration trajectory from DMPs
        # normalize the distance in negative part and positive part
        max_dis = self.config['reward_shaping']['max_EEF_distance']
        dis_negative_low = cus_log(x_value=max_dis + (1 - dis_threshold), base_x=log_base)  # specified, need to test if changed the parameters
        dis_negative_high = cus_log(x_value=dis_threshold + (1 - dis_threshold), base_x=log_base)
        dis_positive_low = cus_log(x_value=dis_threshold + (1 - dis_threshold), base_x=log_base)
        dis_positive_high = cus_log(x_value=0 + (1 - dis_threshold), base_x=log_base)

        # calculater pos and rot err
        rew_rot = 0  # some envs do not need rotation reward
        if self.env_index == 2:  # pour skill, calculate the distance between cube and the area
            cube_pos = self.sim.get_body_position('pourcube')
            rew_pos = euclidean_distance(self.goal, cube_pos)
            # rew_rot = 0  # disable the rotation distance
        else:  # flip and reach skill, calculate the distance between EEF site and target goal, contain rotation
            if self.env_index == 0:
                rew_rot = euler_angle_distance(self.target_rot, self.sim.get_site_euler(self.tool_site, rot_type='zyx'))
                # print('reward info - rot:',rew_rot, self.target_rot, self.sim.get_site_euler(self.tool_site))
                # print('reach rot:', rew_rot)
            if self.env_index == 1:
                obj_euler = self.sim.get_site_euler(self.tool_site, rot_type='zyx')
                obj_euler_z = obj_euler[0] # zyx and get z-axis angle
                rew_rot = (np.cos(obj_euler_z) + 1) / 2  # cosine makes min in [-pi, pi] is zero, max is 2
                # print('flip rot:', rew_rot)
            # print(self.target_rot, self.sim.get_site_euler(self.tool_site), self.sim.get_site_quaternion(self.tool_site))
            ee_site_pos = self.sim.get_site_position(self.tool_site)
            rew_pos = euclidean_distance(self.goal, ee_site_pos)
            # print(ee_site_pos, self.sim.get_site_position('attachment_siteL'))
            # print('-------')
            # if self.env_index == 0:
            #     rew_rot = 0  # disable the rotation distance
        # calculater trajectory err between current state and demonstration's state
        curr_ee_pos = self.sim.get_site_position(self.tool_site)  # get the EEF's pos
        curr_ee_rot = self.sim.get_site_euler(self.tool_site)  # get the EEF's euler
        curr_ee_FT = self.sim.get_ft_sensor('Lforce', 'Ltorque') if self.dmp_force_enable is True else []
        curr_state = np.concatenate((curr_ee_pos, curr_ee_rot, curr_ee_FT))

        # !DMPs traj err, contains pos trajectory and rot trajectory
        rew_pos_traj = euclidean_distance(curr_state[:3], self.dmp_traj[:3, self._temporal])
        self._buffer_traj[:, 0] = curr_state[3:6]
        self._buffer_traj = np.roll(self._buffer_traj, -1, axis=1)
        if self._temporal < self._buffer_traj_size:
            rew_rot_traj = 0
        else:
            rew_rot_traj_x = cosine_distance(self._buffer_traj[0, :],
                                             self.dmp_traj[3, (self._temporal-self._buffer_traj_size):self._temporal])
            rew_rot_traj_y = cosine_distance(self._buffer_traj[1, :],
                                             self.dmp_traj[4, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj_z = cosine_distance(self._buffer_traj[2, :],
                                             self.dmp_traj[5, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj = (rew_rot_traj_x + rew_rot_traj_y + rew_rot_traj_z) / 3
        # !clip the traj
        rew_pos_traj = np.clip(rew_pos_traj, 0, 1)
        rew_rot_traj = np.clip(rew_rot_traj, 0, 1)
        rew_traj = (rew_pos_traj + rew_rot_traj) / 2
        # print('reward - traj pos and rot:', rew_pos_traj, rew_rot_traj)
        # print('reward for pos:', rew_traj)
        # print('current pos:', curr_ee_pos, curr_ee_rot)
        # print('dmps traj:', self.dmp_traj[:, self.follow_dmp_step], self.follow_dmp_step)
        # input()

        th = 0.02  # distance is 2cm for reach skill
        rew_dis = rew_pos + 0.35 * rew_rot
        # !clip the reward
        rew_pos = np.clip(rew_pos, 0, 1)
        rew_rot = np.clip(rew_rot, 0, 1)
        rew_dis = (rew_pos + rew_rot) / 2
        # print(rew_rot, rew_pos)

        # if rew_dis > dis_threshold:  # negative part
        #     rew_dis_norm = _normalization(cus_log(rew_dis + (1 - dis_threshold), base_x=log_base),
        #                                       _min=dis_negative_low, _max=dis_negative_high,
        #                                       range_min=-1, range_max=0)
        # else:
        #     rew_dis_norm = _normalization(cus_log(rew_dis + (1 - dis_threshold), base_x=log_base),
        #                                       _min=dis_positive_low, _max=dis_positive_high,
        #                                       range_min=0, range_max=1)
        # # print('rew:', rew, cus_log(rew_dis + (1 - dis_threshold), base_x=log_base))
        # rew = weights[0] * rew_dis_norm + weights[1] * _normalization(-rew_traj,
        #                                                               _min=-traj_threshold, _max=0,
        #                                                               range_min=-1, range_max=1)

        rew = -weights[0] * rew_dis + (-weights[1] * rew_traj)
        # print('dis, traj dis:', rew_dis, rew_traj)
        # rew = -weights[0] * cus_log(rew_pos + (1 - th * 1.3), 10)

        # rew = (-weights[0] * (rew_pos + rew_rot)) + \
        #       (-weights[1] * rew_traj) * (self._temporal / self.dmp_max_step)

        # rew -= self._reward_mean
        # rew /= (self._reward_std + 1e-8)
        #
        # self._reward_buffer.append(rew)
        # print(self._reward_mean, self._reward_std, rew)

        # rew += 1 if self.env_index == 0 and rew_pos < 0.02 else 0

        self.reward_item += 1
        if self.reward_item > 2000:  # log the reward
            logging.info(f'reward each 2000 act: object posture distance is {rew_pos} and {rew_rot}, trajectory distance is {rew_traj}')
            self.reward_item = 0
        return rew


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_site_position(self.tool_site))
        # print('obs real leepos:', L_ee_pos)
        # print(L_ee_pos-self._base_pos, self.sim.get_body_position('grab_obj')-self._base_pos)
        # print(self.sim.get_site_euler(self.tool_site), Rotation.from_euler('xyz', self.sim.get_site_euler(self.tool_site)).as_euler('zyx'))
        # # print(euclidean_distance(L_ee_pos, self.sim.get_body_position('grab_obj')))
        # print('------------')
        L_ee_pos = _normalization(L_ee_pos, self.ee_high + self._base_pos, self.ee_low + self._base_pos, range_max=self.norm_max, range_min=self.norm_min)
        L_ee_quat = np.copy(self.sim.get_site_quaternion(self.tool_site))
        # print('obs real leequat:', L_ee_quat)
        L_ee_quat = _normalization(L_ee_quat, _max=1, _min=-1, range_max=self.norm_max,
                                  range_min=self.norm_min)  # hard code for normalization of quaternion
        L_ee_rot = np.copy(self.sim.get_site_euler(self.tool_site))
        # print(np.rad2deg(L_ee_rot))
        # L_ee_rot = _normalization(L_ee_rot, _max=np.pi, _min=-np.pi, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of quaternion
        L_ee_vels = np.copy(self.sim.get_site_euler(self.tool_site))  # TODO: no site velocity, modify it
        L_ee_vels = _normalization(L_ee_vels, _max=1, _min=-1, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of velocity
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        self._obs_last_ft = lowpass_filter(self._obs_last_ft, L_FT_sensor, 0.7)
        # print(self._obs_last_ft)
        L_FT_sensor = _normalization(self._obs_last_ft, _max=100, _min=-100, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of FT sensor

        # embedding the current time of DMPs' trajectory to observation space as reference trajectory
        next_reference = self._temporal + 1
        if next_reference == self.dmp_max_step:
            next_reference = self.dmp_max_step - 1
        dmp_pos = self.dmp_traj[0:3, next_reference]
        # print('dmp pos:', dmp_pos)
        # input()
        dmp_pos = _normalization(dmp_pos, self.ee_high + self._base_pos, self.ee_low + self._base_pos, range_max=self.norm_max, range_min=self.norm_min)
        dmp_rot = self.dmp_traj[3:6, next_reference]
        dmp_quat = Rotation.from_euler('zyx', dmp_rot, degrees=False).as_quat()
        # print('dmp quat:', dmp_quat, next_reference)
        dmp_quat = _normalization(dmp_quat, _max=1, _min=-1, range_max=self.norm_max,
                                  range_min=self.norm_min)  # hard code for normalization of quaternion
        dmp_rot = _normalization(dmp_rot, _max=np.pi, _min=-np.pi, range_max=self.norm_max, range_min=self.norm_min)
        _item = next_reference / self.dmp_max_step



        # obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat, norm_L_ee_vels, norm_L_FT_snesor, self.dmp_w])  # put the DMPs weights to observation space
        # obs = np.concatenate([L_ee_pos, L_ee_rot, L_FT_sensor, dmp_pos, dmp_rot, [_item, self._ik_active]])
        # print(self.sim.get_site_position(self.tool_site), self.dmp_traj[0:3, next_reference])
        obs = np.concatenate([L_ee_pos, L_ee_quat, dmp_pos, dmp_quat, [_item]])  # F/T sensor change too much in observation space 22-dim
        # obs = np.concatenate([L_ee_pos, L_ee_quat, dmp_pos, dmp_quat, L_FT_sensor, [_item]])  # 28 dim
        # obs = np.concatenate([L_ee_pos, L_ee_quat])  # F/T sensor change too much in observation space
        # obs = np.concatenate([L_ee_pos, L_ee_rot, L_FT_sensor, [self._ik_active]])  # F/T sensor change too much in observation space
        return obs

    def reset(self, index, target_goal) -> bool:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        logging.info('reset')
        # reset timestep in robot
        self.sim.set_timestep(self.config['robot']['mj_timestep'])
        # update reward scaling factor by reward buffer
        if self._reward_buffer:
            # print(self._reward_buffer)
            self._reward_mean, self._reward_std = reward_rescaling(self._reward_buffer)
        self._reward_buffer = []  # reset reward buffer for next epoch
        self._ik_active = 1
        self.truncated_num = 0  # reset the count fot step
        self._temporal = 0  # reset the temporal counter for observation
        self.goal = target_goal
        self.env_index = index
        self.sim.set_forward()
        _reset_goal = False
        # hard code for different skill's environment
        local_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
        start_pos = np.zeros(3)
        start_rot = np.zeros(3)
        target_rot = np.zeros(3)
        base = self.sim.get_body_position('baseL')
        self._base_pos = np.copy(self.sim.get_body_position('baseL'))
        if self.env_index == 0:  # reach skill,
            qpos_random = self.init_qpos
            self.sim.set_joint_qpos(self.joint_list[:-1], qpos_random)  # set the joint randomly
            self.sim.control_joints(self.actuator_list[:-1], qpos_random)
            # testing in the reach env for dmps
            # start_pos, start_quat = self.sim.forward_kinematics_kdl(qpos_random)  # get the reset pos and rot\
            self.last_action_ee_pos, last_action_ee_rot = self.sim.forward_kinematics_ikfast(self.init_qpos)
            self.last_action_ee_pos += base
            start_pos = np.copy(self.last_action_ee_pos)
            self.last_action_ee_rot = Rotation.from_quat(last_action_ee_rot).as_euler('zyx', degrees=False)
            start_rot = np.copy(self.last_action_ee_rot)
            # start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
            # self.target_rot = np.deg2rad(np.random.uniform([-175, -10, -10], [-165, 10, 10])) # for euler xyz
            self.target_rot = np.deg2rad(np.random.uniform([-10, -10, -175], [10, 10, -165])) # for euler zyx
            data_path = local_path + '../../datasets/reach/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)


        elif self.env_index == 1:  # flip skill, use IK to generate one posture, fix the Z-rotation face to the ground
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            sim_euler = np.random.uniform(np.deg2rad([-2, -80, -100]), np.deg2rad([2, 80, -80])) # z y x
            # !given euler x -> y -> z, need to be transferred to z - y - x
            sim_euler_trans = Rotation.from_euler('zyx', sim_euler, degrees=False).as_euler('xyz', degrees=False) # one more code for clarify the euler issue
            sim_quat = Rotation.from_euler('xyz', sim_euler_trans, degrees=False).as_quat()  # rotation convertor
            # q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
            q_inv = self.sim.inverse_kinematics_ikfast(target_position=target_goal - self.sim.get_body_position('baseL') + ee_noise,
                                                       target_orientation=sim_quat,
                                                       q_guess=self.init_qpos)
            inv_done = False
            sample_times = 0

            self.target_rot = sim_euler
            self.target_rot[-1] = np.deg2rad(180)
            self.target_rot = Rotation.from_euler('zyx', self.target_rot, degrees=False).as_euler('xyz', degrees=False)

            data_path = local_path + '../../datasets/flip/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)


            while inv_done is False:
                if q_inv is None:
                    sample_times += 1
                    sim_euler = np.random.uniform(np.deg2rad([-2, -80, -100]), np.deg2rad([2, 80, -80])) # z y x
                    sim_euler_trans = Rotation.from_euler('zyx', sim_euler, degrees=False).as_euler('xyz', degrees=False)
                    sim_quat = Rotation.from_euler('xyz', sim_euler_trans, degrees=False).as_quat()  # rotation convertor
                    q_inv = self.sim.inverse_kinematics_ikfast(
                        target_position=target_goal - self.sim.get_body_position('baseL') + ee_noise,
                        target_orientation=sim_quat,
                        q_guess=self.init_qpos)
                    if sample_times > 500:
                        _reset_goal = True
                        # print(target_goal)
                        # print('break')
                        break
                else:
                    inv_done = True
                    self.q_inv = q_inv
                    # print(q_inv, sim_euler_trans, target_goal - base)
                    self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
                    self.sim.control_joints(self.actuator_list[:-1], q_inv)

                    # testing in the reach env for dmps
                    self.last_action_ee_pos, last_action_ee_rot = self.sim.forward_kinematics_ikfast(q_inv)
                    self.last_action_ee_pos += base
                    start_pos = np.copy(self.last_action_ee_pos)
                    self.last_action_ee_rot = Rotation.from_quat(last_action_ee_rot).as_euler('zyx', degrees=False)
                    # start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
                    start_rot = np.copy(self.last_action_ee_rot)
                    self.target_rot = sim_euler
                    self.target_rot[0] = np.deg2rad(180)
                    # self.target_rot = Rotation.from_euler('zyx', self.target_rot, degrees=False).as_euler('xyz',
                    #                                                                                       degrees=False)
                    # input()




        elif self.env_index == 2:  # pouring skill, set the position of the ee upon the round of fixed area
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
            sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
            target_goal[-1] += 0.3
            q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
            inv_done = False
            sample_times = 0

            data_path = local_path + '../../datasets/pour/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)

            while inv_done is False:
                if q_inv.max() > np.pi or q_inv.min() < -np.pi:
                    sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
                    # sim_euler = np.random.uniform(np.deg2rad([-95, 0, 0]), np.deg2rad([-85, 10, 10]))
                    sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
                    q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
                    # print('change')
                    sample_times += 1
                    if sample_times > 500:
                        _reset_goal = True
                        # print(target_goal)
                        # print('break')
                        break
                else:
                    inv_done = True
                    # print(q_inv, sim_euler, target_goal - base)
                    self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
                    self.sim.control_joints(self.actuator_list[:-1], q_inv)
                    self.sim.set_forward()
                    cube_pos = self.sim.get_body_position('bowl')
                    cube_pos[-1] += 0.1
                    self.sim.set_mocap_pos(mocap='pourcube', pos=cube_pos)

                    # testing in the reach env for dmps
                    start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
                    start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
                    target_rot = sim_euler
                    target_rot[0] += 180


        # the general part of DMPs' parameters
        # print('start pos:', start_pos, 'base:', base)
        # start_pos = start_pos + base
        target_pos = target_goal  # get the reset target goal for dmp
        start_vels, target_vels = np.zeros(6), np.zeros(6)  # do not use velocity now
        start_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        target_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        self.start_state = np.concatenate((start_pos, start_rot, start_forcetorque))
        # print('start pos calculated:', start_pos)
        # input()
        self.target_state = np.concatenate((target_pos, self.target_rot, target_forcetorque))
        demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
            data_path=data_path,
            ex_length=self.dmp_max_step,
        )
        # ! for euler zyx
        demo_ee_rot = Rotation.from_euler('xyz', demo_ee_rot.T).as_euler('zyx').T
        if self.dmp_force_enable is True:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot, demo_eeft), axis=0)
        else:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot), axis=0)
        self.DMPs = dmps.dmp_discrete_dyn_weight(n_dmps=demonstration_trajs.shape[0],
                                                 n_bfs=self.dmp_n_bfs,
                                                 dt=1.0 / demonstration_trajs.shape[1])
        dmp_w = self.DMPs.learning(demonstration_trajs, plot=False)
        self.dmp_w = _normalization(dmp_w.flatten(), dmp_w.flatten().max(), dmp_w.flatten().min(), range_max=1, range_min=-1)

        self.dmp_traj, _, _ = self.DMPs.reproduce(dyn_w_gate=False,
                                                  initial=self.start_state,
                                                  goal=self.target_state,
                                                  )  # get output of dmps, do not need the gradient
        # draw demo trajectory ------
        # self.sim.modify_scene(self.dmp_traj[:3, :])
        # print('--', target_pos, self.goal, start_pos)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(demonstration_trajs[0, :], demonstration_trajs[1, :], demonstration_trajs[2, :])
        # ax.plot(self.dmp_traj[0, :], self.dmp_traj[1, :], self.dmp_traj[2, :])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        # --------------------------
        # simulator forward
        self.sim.set_forward()
        return _reset_goal



import rtde_receive
import rtde_control
from ur_ikfast import ur_kinematics

class singleTool_UR5e_real(MJRobot):
    with open('config/chef_v2.yml', 'r', encoding='utf-8') as cfg:
        config = yaml.load(cfg, Loader=yaml.FullLoader)
    def __init__(self,
                 sim,
                 control_type: str = 'ee',
                 control_ee_rot: bool = True,
                 dmps_weights_act: bool = True,
                 dmps_force_enable: bool = False,
                 control_finger: bool = False,
                 normalization_range: list = [0, 1],
                 ) -> None:
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
        # action space definition
        n_actions = 3 if control_type == 'ee' else 6
        n_actions += 3 if control_ee_rot is True else 0
        n_actions += 2 if control_finger is True else 0
        action_space = spaces.Box(self.norm_min, self.norm_max, shape=(n_actions,), dtype=np.float32)
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.tool_site = 'EEFee_pos'
        self.env_index = -1
        self.ee_high = np.array(self.config['robot']['ee_pos_limitation_high'])
        self.ee_low = np.array(self.config['robot']['ee_pos_limitation_low'])
        self.ee_rot_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high'])
        self.ee_rot_low = np.deg2rad(self.config['robot']['ee_rot_limitation_low'])
        self.ee_rot_flip_high = np.deg2rad(self.config['robot']['ee_rot_limitation_high_flip'])
        self.ee_rot_flip_low = np.deg2rad(self.config['robot']['ee_rot_limitaion_low_flip'])
        self.goal = np.zeros(3)
        self._base_pos = np.zeros(3)
        self.max_step_one_episode = self.config['max_step_one_episode']
        # DMPs configuration
        self.dmp_max_step = self.config['demonstration_length'] # the DMPs length, should plus with the ratio for action step
        self.dmp_x = 0
        self.dmp_n_bfs = self.config['DMPs_weights_num']
        self.dmp_w = np.zeros(self.dmp_n_bfs * 12) if dmps_force_enable is True else np.zeros(self.dmp_n_bfs * 6)  # hard code for shape the size of dmps weight
        self.dmp_force_enable = dmps_force_enable  # the flag for enable force torque DMPs trajectory, delete or add in function
        self.follow_dmp_step = 0  # the step of following DMPs trajectory

        # self.ee_pos_increment_range = np.ones(3) * 0.015  # the maximum action step for EEF's pos, for limit the DMPs' random weight
        # self.ee_rot_increment_range = np.ones(3) * np.deg2rad(5)  # the maximum action step for EEF's rot, for limit the DMPs' random weight
        # self.ee_force_increment_range = np.ones(6) * 10 if dmps_force_enable is True else []
        # self.ee_increment_range = np.concatenate((self.ee_pos_increment_range, self.ee_rot_increment_range, self.ee_force_increment_range))
        # admittance controller configuration, hard code of configuration, using critical damping
        self.adm_controller = AdmController(m=0.5, k=1000, kr=5, dt=0.01)
        self.admittance_params = np.zeros((3, 3))  # contains acc, vel and pos in xyz derictions
        self.admittance_paramsT = np.zeros((3, 3))
        self.truncation_num = 4  # Number of digits to be truncated, used when set_action
        self.last_ft = np.zeros(6)  # the last force torque sensor for guiding admittance controller, used be filtered
        self._obs_last_ft = np.zeros(6)  # for using low-pass filter in observation space for F/T sensor

        self.truncated_num = 0   # record the step item for stop the epoch
        self._temporal = 0  # record the step item for DMPs in observation
        self._ik_active = 1  # the flag for observation to get the ik is working or not
        self.reward_item = 0
        self._reward_std = 1  # for r_t/self._reward_std to get the scaling reward
        self._reward_mean = 0 # for r_t - self._reward_mean to get the scaling reward
        self._reward_buffer = []  # record the one epoch reward for get new standard variation reward

        # record buffer, for reward calculate the distance between shape
        self._buffer_traj_size = 200
        self._buffer_traj = np.zeros((3, self._buffer_traj_size))

        # the setting for real UR5e robot single arm with ikfast func
        self.ur5e_arm = ur_kinematics.URKinematics('ur5e')
        self._ur5e_arm_base_rot = -180  # degree for transfer the base in urdf for ikfast
        self._tool_len_in_y_axis = 0.173  # length of the EEF, configuration for the ikfast
        #### mainly used ikfast function:
        # self.ur5e_arm.forward(q, base_rot, tool_len) # the q is 6 joints in rad
        # self.ur5e_arm.inverse(eepos, base_rot, tool_len) # the eepos is xyz+xyzw
        ####


        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            init_qpos=np.deg2rad([90, -135, 90, -90, -90, 0]),  # hard code for the robot, initial posture
            # init_qpos=np.deg2rad([0, 0, 0, 0, 0, 0]),  # hard code for the robot, initial posture
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            actuator_list=['shoulder_panL', 'shoulder_liftL', 'elbowL', 'wrist_1L', 'wrist_2L', 'wrist_3L', 'fingersL'],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray):
        truncated = False  # for stop the episode when get max step
        # fix the action
        # action = action.copy()
        # act_len = len(action)

        if self.env_index == 2:
            increment_ee_pos = action[:3] * self.config['robot']['ee_pos_increment_pour']
            increment_ee_rot = action[3:] * np.deg2rad(self.config['robot']['ee_rot_increment_pour'])
        else:
            increment_ee_pos = action[:3] * self.config['robot']['ee_pos_increment']
            increment_ee_rot = action[3:] * np.deg2rad(self.config['robot']['ee_rot_increment'])

        # the limited action is set_dmps_traj (12-dim), which is used as the desired state for admittance controller
        des_pos = np.around(self.last_action_ee_pos + increment_ee_pos,
                            self.truncation_num)  # getting pos, rot and force/torque

        des_euler = np.around(self.last_action_ee_rot + increment_ee_rot, self.truncation_num)
        # print('init pos:', increment_ee_pos, self.last_action_ee_pos, des_pos)
        des_pos = np.clip(des_pos, self.ee_low, self.ee_high)
        if self.env_index == 0:
            des_euler = np.clip(des_euler, self.ee_rot_low, self.ee_rot_high)
        elif self.env_index == 1:
            des_euler = np.clip(des_euler, self.ee_rot_flip_low, self.ee_rot_flip_high)
        # des_euler = np.deg2rad([30, 90, -100])
        # print('fixed des_euler in set_action')
        des_quat = Rotation.from_euler('zyx', des_euler, degrees=False).as_quat()
        des_quat = np.roll(des_quat, 1)
        # print(des_pos, des_euler, des_quat)
        self.sim.set_mocap_pos('LEEF', des_pos)
        self.sim.set_mocap_quat('LEEF', des_quat)

        self.last_action_ee_pos = np.copy(des_pos)
        self.last_action_ee_rot = np.copy(des_euler)

        ee_quat = Rotation.from_euler('zyx', des_euler, degrees=False).as_quat(scalar_first=False)

        self._temporal += 1
        if self._temporal > self.dmp_max_step - 1:
            self._temporal = self.dmp_max_step - 1
        self.truncated_num += 1
        self.sim.step()

        ik_ee_pos = np.concatenate((self.last_action_ee_pos, ee_quat))

        q_inv = self.ur5e_arm.inverse(ee_pose=ik_ee_pos,
                                      base_rot=self._ur5e_arm_base_rot,
                                      tool_len_in_y_axis=self._tool_len_in_y_axis,
                                      q_guess=self.ikfast_init_qpos,
                                      )
        print(q_inv)
        if q_inv is not None:
            self.ikfast_init_qpos = np.copy(q_inv)

    def compute_reward(self):
        """
        compute the reward for environment, distance for example
        For different task/skill, the unified reward type would be used:
            Trajectory's distance + Goal distance
            (because we do not know the goal distance for different skill, we get the input directly rather function)
        Returns:
            Float type distance for goal task
        """
        weights = self.config['reward_shaping']['weights']  # the weight for all reward factors, index 0: pos and rot err, index 1: trajectory err
        log_base = self.config['reward_shaping']['log_base']  # logarithm configuration (0<log_base<1)
        dis_threshold = self.config['reward_shaping']['dis_threshold']  # if the distance lower than dis_threshold (like 10 cm, 0.1), the reward go to positive
        traj_threshold = self.config['reward_shaping']['max_traj_diff']  # same function as dis_threshold for demonstration trajectory from DMPs
        # normalize the distance in negative part and positive part
        max_dis = self.config['reward_shaping']['max_EEF_distance']
        dis_negative_low = cus_log(x_value=max_dis + (1 - dis_threshold), base_x=log_base)  # specified, need to test if changed the parameters
        dis_negative_high = cus_log(x_value=dis_threshold + (1 - dis_threshold), base_x=log_base)
        dis_positive_low = cus_log(x_value=dis_threshold + (1 - dis_threshold), base_x=log_base)
        dis_positive_high = cus_log(x_value=0 + (1 - dis_threshold), base_x=log_base)

        # calculater pos and rot err
        rew_rot = 0  # some envs do not need rotation reward
        if self.env_index == 2:  # pour skill, calculate the distance between cube and the area
            cube_pos = self.sim.get_body_position('rigid_cube')
            rew_pos = euclidean_distance(self.goal, cube_pos)
            # print('rew dis:', rew_pos, self.goal, cube_pos)
            # rew_rot = 0  # disable the rotation distance
        else:  # flip and reach skill, calculate the distance between EEF site and target goal, contain rotation
            if self.env_index == 0:
                rew_rot = euler_angle_distance(self.target_rot, self.sim.get_site_euler(self.tool_site, rot_type='zyx'))
                # print(rew_rot)
            # print(self.target_rot, self.sim.get_site_euler(self.tool_site), self.sim.get_site_quaternion(self.tool_site))
            ee_site_pos = self.sim.get_site_position(self.tool_site)
            pos_dis = euclidean_distance(self.goal, ee_site_pos)
            # print('pos dis:', pos_dis)
            rew_pos = pos_dis
            # if self.env_index == 0:
            #     rew_rot = 0  # disable the rotation distance
            if self.env_index == 1:
                obj_euler = self.sim.get_body_euler('grab_obj', euler_dire='zyx')
                # print('obj rot:', np.rad2deg(obj_euler))
                # rew_rot = euler_angle_distance(self.target_rot, obj_euler)  # get the reward of rotation, only focus on x-aixs and y-axis
                rew_rot = abs(np.cos(obj_euler[0]) - np.cos(self.target_rot[0])) / 2
        # calculater trajectory err between current state and demonstration's state
        curr_ee_pos = self.sim.get_site_position(self.tool_site)  # get the EEF's pos
        curr_ee_rot = self.sim.get_site_euler(self.tool_site)  # get the EEF's euler
        curr_state = np.concatenate((curr_ee_pos, curr_ee_rot))

        rew_pos_traj = euclidean_distance(curr_state[:3], self.dmp_traj[:3, self._temporal])
        # print(rew_pos_traj)
        self._buffer_traj[:, 0] = curr_state[3:6]
        self._buffer_traj = np.roll(self._buffer_traj, -1, axis=1)
        if self._temporal < self._buffer_traj_size:
            rew_rot_traj = 0
        else:
            rew_rot_traj_x = cosine_distance(self._buffer_traj[0, :],
                                             self.dmp_traj[3, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj_y = cosine_distance(self._buffer_traj[1, :],
                                             self.dmp_traj[4, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj_z = cosine_distance(self._buffer_traj[2, :],
                                             self.dmp_traj[5, (self._temporal - self._buffer_traj_size):self._temporal])
            rew_rot_traj = (rew_rot_traj_x + rew_rot_traj_y + rew_rot_traj_z) / 3
            # print(rew_rot_traj)
        rew_traj = rew_pos_traj + rew_rot_traj
        # rew_traj = rew_pos_traj
        # print('reward - traj pos and rot:', rew_pos_traj, rew_rot_traj)
        # print('current pos:', curr_ee_pos, curr_ee_rot)
        # print('dmps traj:', self.dmp_traj[:, self.follow_dmp_step], self.follow_dmp_step)
        # input()

        th = 0.02  # distance is 2cm for reach skill
        rew_dis = rew_pos + 0.1 * rew_rot
        # print('reward:', rew_pos, rew_rot)
        # norm method
        # rew_dis = rew_pos + rew_rot
        if rew_dis > dis_threshold:  # negative part
            rew_dis_norm = _normalization(cus_log(rew_dis + (1 - dis_threshold), base_x=log_base),
                                              _min=dis_negative_low, _max=dis_negative_high,
                                              range_min=-1, range_max=0)
        else:
            rew_dis_norm = _normalization(cus_log(rew_dis + (1 - dis_threshold), base_x=log_base),
                                              _min=dis_positive_low, _max=dis_positive_high,
                                              range_min=0, range_max=1)
        # print('rew:', rew, cus_log(rew_dis + (1 - dis_threshold), base_x=log_base))

        # rew = weights[0] * rew_dis_norm + \
        #       weights[1] * _normalization(-rew_traj,
        #                                   _min=-traj_threshold, _max=0,
        #                                   range_min=-1, range_max=1)
        # rew = -weights[0] * rew_dis

        # rew = weights[0] * rew_dis_norm



        # rew = -weights[0] * cus_log(rew_pos + (1 - th * 1.3), 10)

        rew = (-weights[0] * rew_dis) + \
              (-weights[1] * rew_traj)

        # rew -= self._reward_mean
        # rew /= (self._reward_std + 1e-8)
        #
        # self._reward_buffer.append(rew)
        # print(self._reward_mean, self._reward_std, rew)

        # rew += 1 if self.env_index == 0 and rew_pos < 0.02 else 0

        self.reward_item += 1
        if self.reward_item > 2000:  # log the reward
            logging.info(f'reward each 2000 act: object posture distance is {rew_pos} and {rew_rot}, trajectory distance is {rew_traj}')
            self.reward_item = 0
        return rew


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_site_position(self.tool_site))
        # print('LEEPOS:', L_ee_pos)
        # print(L_ee_pos-self._base_pos, self.sim.get_body_position('grab_obj')-self._base_pos)
        # print(self.sim.get_site_euler(self.tool_site))
        # print(euclidean_distance(L_ee_pos, self.sim.get_body_position('grab_obj')))
        # print('------------')
        L_ee_pos = _normalization(L_ee_pos, self.ee_high, self.ee_low, range_max=self.norm_max, range_min=self.norm_min)
        L_ee_quat = np.copy(self.sim.get_site_quaternion(self.tool_site))
        L_ee_rot = np.copy(self.sim.get_site_euler(self.tool_site, rot_type='zyx'))
        # print('robot rot:', np.rad2deg(L_ee_rot))
        L_ee_rot = _normalization(L_ee_rot, _max=self.ee_rot_high, _min=self.ee_rot_low, range_max=self.norm_max, range_min=self.norm_min)  # hard code for normalization of quaternion
        # embedding the current time of DMPs' trajectory to observation space as reference trajectory
        next_reference = self._temporal + 1
        if next_reference == self.dmp_max_step:
            next_reference = self.dmp_max_step - 1
        dmp_pos = self.dmp_traj[0:3, next_reference]
        dmp_pos = _normalization(dmp_pos, self.ee_high, self.ee_low, range_max=self.norm_max, range_min=self.norm_min)
        dmp_rot = self.dmp_traj[3:6, next_reference]
        # print('traj rot:', np.rad2deg(dmp_rot))
        dmp_quat = Rotation.from_euler('zyx', dmp_rot, degrees=False).as_quat()
        dmp_rot = _normalization(dmp_rot, _max=np.pi, _min=-np.pi, range_max=self.norm_max, range_min=self.norm_min)
        _item = next_reference / self.dmp_max_step
        # print(_item)
        # obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat, norm_L_ee_vels, norm_L_FT_snesor, self.dmp_w])  # put the DMPs weights to observation space
        # obs = np.concatenate([L_ee_pos, L_ee_rot, L_FT_sensor, dmp_pos, dmp_rot, [_item, self._ik_active]])
        # print(self.sim.get_site_position(self.tool_site), self.dmp_traj[0:3, next_reference])
        obs = np.concatenate([L_ee_pos, L_ee_quat, dmp_pos, dmp_quat, [_item]])  # F/T sensor change too much in observation space
        # obs = np.concatenate([L_ee_pos, L_ee_quat])
        return obs

    def reset(self, index, target_goal) -> bool:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        ## init robot when task start
        rtde_r = rtde_receive.RTDEReceiveInterface("10.42.0.162")
        self.ikfast_init_qpos = np.array([2.4480130672454834, -1.447563813333847, 1.697779957448141, -1.8043166599669398, -1.5562823454486292, 4.04017972946167])
        ee_posture_real = self.ur5e_arm.forward(joint_angles=self.ikfast_init_qpos,
                                                base_rot=self._ur5e_arm_base_rot,
                                                tool_len_in_y_axis=self._tool_len_in_y_axis)
        print('ikfast (px, py, pz, rx, ry ,rz, w):', ee_posture_real)
        # input()
        #------------------------------

        logging.info('reset')
        # update reward scaling factor by reward buffer
        if self._reward_buffer:
            # print(self._reward_buffer)
            self._reward_mean, self._reward_std = reward_rescaling(self._reward_buffer)
        self._reward_buffer = []  # reset reward buffer for next epoch
        self._ik_active = 1
        self.truncated_num = 0  # reset the count fot step
        self._temporal = 0  # reset the temporal counter for observation
        self.goal = target_goal
        self.env_index = index
        self.sim.set_forward()
        _reset_goal = False
        # hard code for different skill's environment
        local_path = os.path.dirname(os.path.abspath(__file__)) + os.path.sep
        start_pos = np.zeros(3)
        start_rot = np.zeros(3)
        target_rot = np.zeros(3)

        if self.env_index == 0:  # reach skill,
            # testing in the reach env for dmps
            # start_pos, start_quat = self.sim.forward_kinematics_kdl(qpos_random)  # get the reset pos and rot
            # get ee pos (start and end)
            # init_pos = np.random.uniform([-0.05, -0.7, 0.3], [0.75, 0.1, 0.6])
            init_posture = self.ur5e_arm.forward(joint_angles=self.ikfast_init_qpos, base_rot=self._ur5e_arm_base_rot, tool_len_in_y_axis=self._tool_len_in_y_axis)
            init_pos = init_posture[:3]
            self.last_action_ee_pos = init_pos
            self.sim.set_mocap_pos('LEEF', self.last_action_ee_pos)  # reset to the init posture
            start_pos = np.copy(self.last_action_ee_pos)
            # get ee rot (start and end)
            # self.last_action_ee_rot = np.deg2rad(np.random.uniform(self.config['robot']['ee_rot_limitation_low'], self.config['robot']['ee_rot_limitation_high']))
            self.last_action_ee_rot = Rotation.from_quat(init_posture[3:], scalar_first=False).as_euler('zyx', degrees=False)
            self.sim.set_mocap_quat('LEEF', Rotation.from_euler('zyx', self.last_action_ee_rot).as_quat())
            start_rot = np.copy(self.last_action_ee_rot)
            # print(start_rot)
            # start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
            minus = np.random.uniform(-1, 1)
            if minus <= 0:
                self.target_rot = np.deg2rad(np.random.uniform([-15, -15, -175], [15, 15, -165]))
            else:
                self.target_rot = np.deg2rad(np.random.uniform([-15, -15, -95], [15, 15, -85]))
            # self.target_rot = np.deg2rad(np.random.uniform([-10, -10, -95], [10, 10, -85]))
            data_path = local_path + '../../datasets/reach/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)
            print(data_path)


        elif self.env_index == 1:  # flip skill, use IK to generate one posture, fix the Z-rotation face to the ground
            # get ee pos (start and end)
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            self.last_action_ee_pos = target_goal + ee_noise
            self.sim.set_mocap_pos('LEEF', self.last_action_ee_pos)
            start_pos = np.copy(self.last_action_ee_pos)
            self.sim.set_mocap_pos('virtual_goal', target_goal)
            # get ee rot (start and end)
            self.last_action_ee_rot = np.random.uniform(np.deg2rad([-2, -80, -100]), np.deg2rad([2, 80, -80]))
            self.sim.set_mocap_quat('LEEF', Rotation.from_euler('zyx', self.last_action_ee_rot).as_quat())
            start_rot = np.copy(self.last_action_ee_rot)

            minus = np.random.uniform(-1, 1)
            if minus <= 0:
                self.target_rot = np.deg2rad(np.random.uniform([-175, -80, -100], [-165, 80, -80]))
            else:
                self.target_rot = np.deg2rad(np.random.uniform([165, -80, -100], [175, 80, -80]))

            data_path = local_path + '../../datasets/flip/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)
            print(data_path)

        elif self.env_index == 2:  # pouring skill, set the position of the ee upon the round of fixed area
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            self.last_action_ee_pos = target_goal + ee_noise
            self.last_action_ee_pos[-1] += 0.55
            self.sim.set_mocap_pos('LEEF', self.last_action_ee_pos)
            start_pos = np.copy(self.last_action_ee_pos)

            self.last_action_ee_rot = np.random.uniform(np.deg2rad([-80, -5, -175]), np.deg2rad([80, 5, -165]))
            self.sim.set_mocap_quat('LEEF', Rotation.from_euler('zyx', self.last_action_ee_rot).as_quat())
            start_rot = np.copy(self.last_action_ee_rot)

            self.sim.set_forward()
            # set the cube init pos
            cube_pos = np.copy(self.sim.get_body_position('bowl'))
            # print(cube_pos, self.last_action_ee_pos)
            cube_pos[-1] -= 0.15
            self.sim.set_mocap_pos('pourcube', cube_pos)

            self.sim.set_forward()
                # self.sim.step()

            minus = np.random.uniform(-1, 1)
            if minus <= 0:
                self.target_rot = np.deg2rad(np.random.uniform([-100, -100, -175], [-80, -80, -165]))
            else:
                self.target_rot = np.deg2rad(np.random.uniform([80, 80, -175], [100, 100, -165]))

            data_path = local_path + '../../datasets/pour/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)
            print(data_path)
            self.sim.set_forward()
        #     data_path = local_path + '../../datasets/pour/'
        #     data_names = os.listdir(data_path)
        #     data_path = data_path + random.choice(data_names)
        #
        #     while inv_done is False:
        #         if q_inv.max() > np.pi or q_inv.min() < -np.pi:
        #             sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
        #             # sim_euler = np.random.uniform(np.deg2rad([-95, 0, 0]), np.deg2rad([-85, 10, 10]))
        #             sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
        #             q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
        #             # print('change')
        #             sample_times += 1
        #             if sample_times > 500:
        #                 _reset_goal = True
        #                 # print(target_goal)
        #                 # print('break')
        #                 break
        #         else:
        #             inv_done = True
        #             # print(q_inv, sim_euler, target_goal - base)
        #             self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
        #             self.sim.control_joints(self.actuator_list[:-1], q_inv)
        #             self.sim.set_forward()
        #             cube_pos = self.sim.get_body_position('bowl')
        #             cube_pos[-1] += 0.1
        #             self.sim.set_mocap_pos(mocap='pourcube', pos=cube_pos)
        #
        #             # testing in the reach env for dmps
        #             start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
        #             start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
        #             target_rot = sim_euler
        #             target_rot[0] += 180


        # the general part of DMPs' parameters
        # print('start pos:', start_pos, 'base:', base)
        # start_pos = start_pos + base
        if self.env_index == 0 or self.env_index == 1: # reach, flip
            target_pos = self.goal  # get the reset target goal for dmp
        elif self.env_index == 2: # pour
            target_pos = self.last_action_ee_pos
        start_vels, target_vels = np.zeros(6), np.zeros(6)  # do not use velocity now
        start_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        target_forcetorque = np.zeros(6) if self.dmp_force_enable is True else []
        self.start_state = np.concatenate((start_pos, start_rot, start_forcetorque))
        # print('start pos calculated:', start_pos)
        # input()
        self.target_state = np.concatenate((target_pos, self.target_rot, target_forcetorque))
        demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
            data_path=data_path,
            ex_length=self.dmp_max_step,
        )
        # ! for euler zyx
        demo_ee_rot = Rotation.from_euler('xyz', demo_ee_rot.T).as_euler('zyx').T
        if self.dmp_force_enable is True:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot, demo_eeft), axis=0)
        else:
            demonstration_trajs = np.concatenate((demo_ee_pos, demo_ee_rot), axis=0)
        self.DMPs = dmps.dmp_discrete_dyn_weight(n_dmps=demonstration_trajs.shape[0],
                                                 n_bfs=self.dmp_n_bfs,
                                                 dt=1.0 / demonstration_trajs.shape[1])
        dmp_w = self.DMPs.learning(demonstration_trajs, plot=False)
        self.dmp_w = _normalization(dmp_w.flatten(), dmp_w.flatten().max(), dmp_w.flatten().min(), range_max=1, range_min=-1)

        self.dmp_traj, _, _ = self.DMPs.reproduce(dyn_w_gate=False,
                                                  initial=self.start_state,
                                                  goal=self.target_state,
                                                  )  # get output of dmps, do not need the gradient
        # draw demo trajectory ------
        # self.sim.modify_scene(self.dmp_traj[:3, :])
        # print('--', target_pos, self.goal, start_pos)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(demonstration_trajs[0, :], demonstration_trajs[1, :], demonstration_trajs[2, :])
        # ax.plot(self.dmp_traj[0, :], self.dmp_traj[1, :], self.dmp_traj[2, :])
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        # --------------------------
        # simulator forward
        self.sim.set_forward()
        return _reset_goal

