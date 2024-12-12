import random

import numpy as np
import math
from gymnasium_envs.envs.core import MJRobot
from gymnasium import spaces

from gymnasium_envs.utils import _normalization, interp_preprocessed_data_with_vel, euclidean_distance, lowpass_filter, cosine_distance
from scipy.spatial.transform import Rotation
from gymnasium_envs.DMPs import dmps
from gymnasium_envs.admittance_controller.core import FT_controller as AdmController
import os
import logging
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
        self.ee_high = np.array([0.2, 0.6, 0.8])
        self.ee_low = np.array([-0.2, 0.1, 0.2])
        self.ee_rot_high = np.deg2rad([-125, 10, 10])
        self.ee_rot_low = np.deg2rad([-145, -10, -10])
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
        return truncated
        # zeros_angle = [0, 0, 0, 0, 0, 0]
        # self.sim.set_joint_qpos(self.joint_list[:-1], self.init_qpos)
        # print(self.sim.get_body_position(self.L_eef) - self.sim.get_body_position('baseL'))
        # # ---KDL Test---
        # test_angle_zero = np.zeros(6)
        # test_angle = np.deg2rad([45, 45, 45, 45, 45, 45])
        # # test_angle = [0, 0, 0, 0, 0, -1.57]
        # # test_angle = [0.3235, 0.7235,0.2235,0.4235,0.1235,0.5235]
        # self.sim.set_joint_qpos(self.joint_list[:-1], self.init_qpos)
        # qpos = np.array(test_angle)
        # # print(np.array([self.sim.get_joint_angle(joint=self.joint_list[i]) for i in range(7)]))
        # if self.env_index == 1:
        #     base = self.sim.get_body_position('baseL')
        #     sim_pos = self.sim.get_body_position('grab_obj')
        #     sim_qua = np.roll(self.sim.get_body_quaternion('grab_obj'), -1)
        #     sim_qua_test = [0.3535534, 0, 0.7071068, 0.6123724]
        #     kdl_pos, kdl_qua = self.sim.forward_kinematics_kdl(self.init_qpos)
        #     print(kdl_qua, Rotation.from_quat(kdl_qua).as_euler('xyz', degrees=True))
        #     q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, sim_pos - base,
        #                                             sim_qua)
        #     print(q_inv)
        #
        #     # curr_joint_qpos = np.zeros(6)
        #     # target_pos, target_quat = sim_pos, sim_qua
        #     # start_pos, start_quat = self.sim.forward_kinematics_kdl(self.init_qpos)
        #     # start_pos += base
        #     # print(start_pos, start_quat)
        #     # target_euler = Rotation.from_quat(target_quat).as_euler('xyz', degrees=False)
        #     # start_euler = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
        #     # traj_num = 10
        #     # traj_pos, traj_euler = np.zeros((3, traj_num)), np.zeros((3, traj_num))
        #     # for i in range(3):
        #     #     if target_pos[i] >= start_pos[i]:
        #     #         traj_pos[i, :] = np.linspace(start_pos[i], target_pos[i], traj_num)
        #     #         traj_euler[i, :] = np.linspace(start_euler[i], target_euler[i], traj_num)
        #     #     else:
        #     #         traj_pos[i, :] = np.linspace(target_pos[i], start_pos[i], traj_num)[::-1]
        #     #         traj_euler[i, :] = np.linspace(target_euler[i], start_euler[i], traj_num)[::-1]
        #     #
        #     # q_inv = self.init_qpos
        #     # for i in range(traj_num):
        #     #     traj_quat = Rotation.from_euler('xyz', traj_euler[:, i], degrees=False).as_quat()
        #     #     q_inv = self.sim.inverse_kinematics_kdl(q_inv, traj_pos[:, i] - base,
        #     #                                             traj_quat)
        #     #     # print(q_inv)
        #     #     # print(traj_pos[:, i], traj_quat, start_pos, start_quat)
        #     #     # print('---------')
        #     #     # input()
        #
        #     print('-------------', '\n',
        #         'sim pos:', sim_pos, '\n',
        #         'sim qua:', sim_qua, '\n',
        #         'kdl pos:', kdl_pos + base, '\n',
        #         'kdl qua:', kdl_qua, '\n',
        #         'kdl q inv:', q_inv,
        #     )
        #     input()
        # # ---------------------


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
                rew_rot = cosine_distance(self.target_state[3:5], obj_euler[:2]) # get the reward of rotation, only focus on x-aixs and y-axis

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

class singleUR5e(MJRobot):
    def __init__(self,
                 sim,
                 control_type: str = 'ee',
                 control_ee_rot: bool = True,
                 dmps_weights_num: int = 10,
                 dmps_weights_act: bool = True,
                 dmps_force_enable: bool = False,
                 control_finger: bool = False,
                 normalization_range: list = [0, 1],
                 ) -> None:
        self.sensor_num = 4
        self.norm_max = normalization_range[1]
        self.norm_min = normalization_range[0]
        # action space definition
        n_actions = 3 if control_type == 'ee' else 6
        n_actions += 3 if control_ee_rot is True else 0
        n_actions += 2 if control_finger is True else 0
        n_actions += 6 if dmps_weights_act is True and dmps_force_enable is True else 0  # add the force torque demonstration when using DMPs
        n_actions *= dmps_weights_num if dmps_weights_act is True else 1
        # print(n_actions)
        # input()
        action_space = spaces.Box(self.norm_min, self.norm_max, shape=(n_actions,), dtype=np.float32)
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.env_index = -1
        self.arm_ee_max_pos = np.array([-0.5 + 1, 1, 0.816 + 1.1])
        self.arm_ee_min_pos = np.array([-0.5 - 1, -1, 0.816 - 1.1])
        self.ee_high = np.array([0.2, 0.7, 0.8])
        self.ee_low = np.array([-0.2, 0.1, 0.2])
        self.ee_rot_high = np.deg2rad([-90, 10, 30])
        self.ee_rot_low = np.deg2rad([-180, -10, -30])
        self.goal = np.zeros(3)
        # DMPs configuration
        self.dmp_x = 0
        self.dmp_n_bfs = dmps_weights_num
        self.dmp_w = np.zeros(self.dmp_n_bfs * 12) if dmps_force_enable is True else np.zeros(self.dmp_n_bfs * 6)  # hard code for shape the size of dmps weight
        self.dmp_force_enable = dmps_force_enable  # the flag for enable force torque DMPs trajectory, delete or add in function
        self.follow_dmp_step = 0  # the step of following DMPs trajectory
        self.dmp_max_step = 2000  # hard code, the DMPs length
        self.ee_pos_increment_range = np.ones(3) * 0.01  # the maximum action step for EEF's pos, for limit the DMPs' random weight
        self.ee_rot_increment_range = np.ones(3) * np.deg2rad(5)  # the maximum action step for EEF's rot, for limit the DMPs' random weight
        self.ee_force_increment_range = np.ones(6) * 10 if dmps_force_enable is True else []
        self.ee_increment_range = np.concatenate((self.ee_pos_increment_range, self.ee_rot_increment_range, self.ee_force_increment_range))
        # admittance controller configuration, hard code of configuration, using critical damping
        self.adm_controller = AdmController(m=0.5, k=1000, kr=5, dt=0.01)
        self.admittance_params = np.zeros((3, 3))  # contains acc, vel and pos in xyz derictions
        self.admittance_paramsT = np.zeros((3, 3))
        self.truncation_num = 4  # Number of digits to be truncated, used when set_action
        self.last_ft = np.zeros(6)  # the last force torque sensor for guiding admittance controller, used be filtered

        self.truncated_num = 0

        self.reward_item = 0

        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            init_qpos=np.deg2rad([90, -135, 90, -90, -90, 0]),
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            actuator_list=['shoulder_panL', 'shoulder_liftL', 'elbowL', 'wrist_1L', 'wrist_2L', 'wrist_3L', 'fingersL'],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray):
        truncated = False
        # fix the action
        action = action.copy()
        act_len = len(action)

        # reshape action to the shape of weight for DMPs
        action = action.reshape(int(act_len/self.dmp_n_bfs), self.dmp_n_bfs)

        # get the current time of target EEF action by DMPs (Do NOT get the whole trajectory, too slow for the loop)
        set_dmps_traj, _, _, self.dmp_x= self.DMPs.reproduce_single_point(
                                                                dyn_w_gate=True,
                                                                dyn_w=action,
                                                                norm_range_min=self.norm_min,
                                                                norm_range_max=self.norm_max,
                                                                initial=self.start_state,
                                                                goal=self.target_state,
                                                                time_point=self.follow_dmp_step,  # TODO: the time point need to be controlled
                                                                )
        curr_ee_pos = self.sim.get_site_position('EEFee_pos')  # get the EEF's pos
        curr_ee_rot = self.sim.get_site_euler('EEFee_pos')  # get the EEF's euler
        # self.last_action_ee_pos
        curr_ee_FT = self.sim.get_ft_sensor('Lforce', 'Ltorque') if self.dmp_force_enable is True else []
        curr_state = np.concatenate((curr_ee_pos, curr_ee_rot, curr_ee_FT))
        _error_dmps = set_dmps_traj - curr_state
        # print('error dmps reproduce:', _error_dmps, set_dmps_traj)
        # limit the change of pos, rot, force and torque
        _err_extend_index = np.where((abs(_error_dmps) - self.ee_increment_range) > 0)[0]
        if len(_err_extend_index) > 0:
            # print(_err_extend_index[0], len(_err_extend_index))
            for i in range(len(_err_extend_index)):
                _index = _err_extend_index[i]
                # print('--', _index, i)
                if _error_dmps[_index] < 0:
                    set_dmps_traj[_index] = curr_state[_index] - self.ee_increment_range[_index]
                else:
                    set_dmps_traj[_index] = curr_state[_index] + self.ee_increment_range[_index]

        # the limited action is set_dmps_traj (12-dim), which is used as the desired state for admittance controller
        des_pos = np.around(set_dmps_traj[0:3], self.truncation_num)  # hard code the getting pos, rot and force/torque
        des_euler = np.around(set_dmps_traj[3:6], self.truncation_num)
        des_ft = np.around(set_dmps_traj[6:12], self.truncation_num) if self.dmp_force_enable is True else np.zeros(6)
        # !get current force/toque sensor's data for lowpass fileter
        _curr_ft = np.around(self.sim.get_ft_sensor('Lforce', 'Ltorque'), self.truncation_num)
        # print(des_ft.shape, _curr_ft.shape)
        _err_ft = _curr_ft - des_ft
        self.last_ft = lowpass_filter(self.last_ft, _err_ft)
        pos_d, rot_d, self.admittance_params, self.admittance_paramsT = self.adm_controller.admittance_control(
                                                                                desired_position=des_pos,
                                                                                desired_rotation=des_euler,
                                                                                FT_data=self.last_ft,
                                                                                params_mat=self.admittance_params,
                                                                                paramsT_mat=self.admittance_paramsT,
                                                                                )
        # print(pos_d)
        position_d = np.around(pos_d, self.truncation_num)
        rotation_d = np.around(rot_d, self.truncation_num)
        r_target_quat = Rotation.from_euler('xyz', rotation_d, degrees=False).as_quat()
        curr_qpos = np.array([self.sim.get_joint_angle(joint=self.joint_list[i]) for i in range(6)])  # get the qpos without finger
        ik_qpos = self.sim.inverse_kinematics_kdl(current_joint=curr_qpos,
                                                  target_position=position_d - self.sim.get_body_position('baseL'),
                                                  target_orientation=r_target_quat,)
        ik_qpos = np.around(ik_qpos, self.truncation_num)
        # print('current ee pos:', curr_ee_pos)
        # print('desired ee pos:', position_d)
        # print('current qpos', curr_qpos)
        # print('ik qpos:', ik_qpos)
        # print('------------')
        # input()
        self.sim.control_joints(self.actuator_list[:-1], ik_qpos)
        self.follow_dmp_step += 1
        if self.follow_dmp_step >= self.dmp_max_step:
            self.follow_dmp_step = self.dmp_max_step-1
            self.truncated_num += 1
            if self.truncated_num > 100:  # hard code for early stop / get
                truncated = True
                self.truncated_num = 0
                # print('early stop')
        self.sim.step()
        return truncated
        # zeros_angle = [0, 0, 0, 0, 0, 0]
        # self.sim.set_joint_qpos(self.joint_list[:-1], self.init_qpos)
        # print(self.sim.get_body_position(self.L_eef) - self.sim.get_body_position('baseL'))
        # # ---KDL Test---
        # test_angle_zero = np.zeros(6)
        # test_angle = np.deg2rad([45, 45, 45, 45, 45, 45])
        # # test_angle = [0, 0, 0, 0, 0, -1.57]
        # # test_angle = [0.3235, 0.7235,0.2235,0.4235,0.1235,0.5235]
        # self.sim.set_joint_qpos(self.joint_list[:-1], self.init_qpos)
        # qpos = np.array(test_angle)
        # # print(np.array([self.sim.get_joint_angle(joint=self.joint_list[i]) for i in range(7)]))
        # if self.env_index == 1:
        #     base = self.sim.get_body_position('baseL')
        #     sim_pos = self.sim.get_body_position('grab_obj')
        #     sim_qua = np.roll(self.sim.get_body_quaternion('grab_obj'), -1)
        #     sim_qua_test = [0.3535534, 0, 0.7071068, 0.6123724]
        #     kdl_pos, kdl_qua = self.sim.forward_kinematics_kdl(self.init_qpos)
        #     print(kdl_qua, Rotation.from_quat(kdl_qua).as_euler('xyz', degrees=True))
        #     q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, sim_pos - base,
        #                                             sim_qua)
        #     print(q_inv)
        #
        #     # curr_joint_qpos = np.zeros(6)
        #     # target_pos, target_quat = sim_pos, sim_qua
        #     # start_pos, start_quat = self.sim.forward_kinematics_kdl(self.init_qpos)
        #     # start_pos += base
        #     # print(start_pos, start_quat)
        #     # target_euler = Rotation.from_quat(target_quat).as_euler('xyz', degrees=False)
        #     # start_euler = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
        #     # traj_num = 10
        #     # traj_pos, traj_euler = np.zeros((3, traj_num)), np.zeros((3, traj_num))
        #     # for i in range(3):
        #     #     if target_pos[i] >= start_pos[i]:
        #     #         traj_pos[i, :] = np.linspace(start_pos[i], target_pos[i], traj_num)
        #     #         traj_euler[i, :] = np.linspace(start_euler[i], target_euler[i], traj_num)
        #     #     else:
        #     #         traj_pos[i, :] = np.linspace(target_pos[i], start_pos[i], traj_num)[::-1]
        #     #         traj_euler[i, :] = np.linspace(target_euler[i], start_euler[i], traj_num)[::-1]
        #     #
        #     # q_inv = self.init_qpos
        #     # for i in range(traj_num):
        #     #     traj_quat = Rotation.from_euler('xyz', traj_euler[:, i], degrees=False).as_quat()
        #     #     q_inv = self.sim.inverse_kinematics_kdl(q_inv, traj_pos[:, i] - base,
        #     #                                             traj_quat)
        #     #     # print(q_inv)
        #     #     # print(traj_pos[:, i], traj_quat, start_pos, start_quat)
        #     #     # print('---------')
        #     #     # input()
        #
        #     print('-------------', '\n',
        #         'sim pos:', sim_pos, '\n',
        #         'sim qua:', sim_qua, '\n',
        #         'kdl pos:', kdl_pos + base, '\n',
        #         'kdl qua:', kdl_qua, '\n',
        #         'kdl q inv:', q_inv,
        #     )
        #     input()
        # # ---------------------


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
                rew_rot = cosine_distance(self.target_state[3:5], obj_euler[:2]) # get the reward of rotation, only focus on x-aixs and y-axis

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

        self.reward_item += 1
        if self.reward_item > 500:
            logging.info(f'reward each 500 act: object posture distance is {rew_pos} and  {rew_rot}, trajectory distance is {rew_traj}')
            self.reward_item = 0
        return rew


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_body_position(self.L_eef))
        norm_L_ee_pos = _normalization(L_ee_pos, self.arm_ee_max_pos, self.arm_ee_min_pos, range_max=self.norm_max, range_min=self.norm_min)
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
        obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat, norm_L_ee_vels, norm_L_FT_snesor, [self.dmp_x]])
        return obs

    def reset(self, index, target_goal) -> bool:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
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
            self.sim.set_joint_qpos(self.joint_list[:-1], qpos_random)  # set the joint randomly
            self.sim.control_joints(self.actuator_list[:-1], qpos_random)
            # testing in the reach env for dmps
            # start_pos, start_quat = self.sim.forward_kinematics_kdl(qpos_random)  # get the reset pos and rot
            self.last_action_ee_pos, last_action_ee_rot = self.sim.forward_kinematics_ikfast(self.init_qpos)
            self.last_action_ee_pos += base
            start_pos = np.copy(self.last_action_ee_pos)
            self.last_action_ee_rot = Rotation.from_quat(last_action_ee_rot).as_euler('xyz', degrees=False)
            start_rot = np.copy(self.last_action_ee_rot)
            # start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
            target_rot = np.deg2rad(np.random.uniform([-180, -10, -30], [-90, 10, 30]))
            data_path = local_path + '../../datasets/reach/'
            data_names = os.listdir(data_path)
            data_path = data_path + random.choice(data_names)


        # elif self.env_index == 1:  # flip skill, use IK to generate one posture, fix the Z-rotation face to the ground
        #     ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
        #     sim_euler = np.random.uniform(np.deg2rad([-10, -10, -180]), np.deg2rad([10, 10, 180]))
        #     sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
        #     q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
        #     inv_done = False
        #     sample_times = 0
        #
        #     data_path = local_path + '../../datasets/flip/'
        #     data_names = os.listdir(data_path)
        #     data_path = data_path + random.choice(data_names)
        #
        #     while inv_done is False:
        #         if q_inv.max() > np.pi or q_inv.min() < -np.pi:
        #             sample_times += 1
        #             sim_euler = np.random.uniform(np.deg2rad([-10, -10, -180]), np.deg2rad([10, 10, 180]))
        #             sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
        #             q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
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
        #
        #             # testing in the reach env for dmps
        #             start_pos, start_quat = self.sim.forward_kinematics_kdl(q_inv)  # get the reset pos and rot
        #             start_rot = Rotation.from_quat(start_quat).as_euler('xyz', degrees=False)
        #             target_rot = sim_euler
        #             target_rot[0] = 90
        #
        #
        #
        #
        # elif self.env_index == 2:  # pouring skill, set the position of the ee upon the round of fixed area
        #     ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
        #     sim_euler = np.random.uniform(np.deg2rad([-90, -5, -180]), np.deg2rad([-80, 5, 180]))
        #     sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
        #     target_goal[-1] += 0.3
        #     q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
        #     inv_done = False
        #     sample_times = 0
        #
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

