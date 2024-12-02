import numpy as np
import math
from gymnasium_envs.envs.core import MJRobot
from gymnasium import spaces

from gymnasium_envs.utils import _normalization
from scipy.spatial.transform import Rotation

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


class singleUR5e(MJRobot):
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
        n_actions *= dmps_weights_num if dmps_weights_act is True else 1
        action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype=np.float32)
        # specified site in simulation
        self.L_eef = 'LEEF'
        self.env_index = -1
        self.arm_ee_max_pos = np.array([-0.5 + 0.85, 0.85, 0.816 + 1.1])
        self.arm_ee_min_pos = np.array([-0.5 - 0.85, -0.85, 0.816])

        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            init_qpos=np.deg2rad([0, 0, 90, -90, -90, 0]),
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            actuator_list=['shoulder_panL', 'shoulder_liftL', 'elbowL', 'wrist_1L', 'wrist_2L', 'wrist_3L', 'fingersL'],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray) -> None:
        self.sim.step()
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


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_body_position(self.L_eef))
        norm_L_ee_pos = _normalization(L_ee_pos, self.arm_ee_max_pos, self.arm_ee_min_pos)
        L_ee_quat = np.copy(self.sim.get_body_quaternion(self.L_eef))
        norm_L_ee_quat = _normalization(L_ee_quat, 1, -1)  # hard code for normalization of quaternion
        L_ee_vels = np.copy(self.sim.get_body_velocity(self.L_eef))
        norm_L_ee_vels = _normalization(L_ee_vels, 1, -1)  # hard code for normalization of velocity
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        norm_L_FT_snesor = _normalization(L_FT_sensor, 10, -10)  # hard code for normalization of FT sensor
        obs = np.concatenate([norm_L_ee_pos, norm_L_ee_quat, norm_L_ee_vels, norm_L_FT_snesor,])
        return obs

    def reset(self, index, target_goal) -> bool:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        self.env_index = index
        self.sim.set_forward()
        _reset_goal = False
        # hard code for different skill's environment
        if self.env_index == 0: # reach skill,
            qpos_random = np.deg2rad(np.random.uniform(-np.ones(6) * 45, np.ones(6) * 45))
            self.sim.set_joint_qpos(self.joint_list[:-1], qpos_random) # set the joint randomly
            self.sim.control_joints(self.actuator_list[:-1], qpos_random)
        elif self.env_index == 1: # flip skill, use IK to generate one posture, fix the Z-rotation face to the ground
            base = self.sim.get_body_position('baseL')
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            sim_euler = np.random.uniform(np.deg2rad([-10, -10, 0]), np.deg2rad([10, 10, 180]))
            sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
            q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
            inv_done = False
            sample_times = 0
            while inv_done is False:
                if q_inv.max() > np.pi or q_inv.min() < -np.pi:
                    sample_times += 1
                    sim_euler = np.random.uniform(np.deg2rad([-10, -10, 0]), np.deg2rad([10, 10, 180]))
                    sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
                    q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
                    if sample_times > 500:
                        _reset_goal = True
                        # print(target_goal)
                        print('break')
                        break
                else:
                    inv_done = True
                    # print(q_inv, sim_euler, target_goal - base)
                    self.sim.set_joint_qpos(self.joint_list[:-1], q_inv)
                    self.sim.control_joints(self.actuator_list[:-1], q_inv)
        elif self.env_index == 2:  # pouring skill, set the position of the ee upon the round of fixed area
            base = self.sim.get_body_position('baseL')
            ee_noise = np.random.uniform(np.ones(3) * -0.02, np.ones(3) * 0.02)
            sim_euler = np.random.uniform(np.deg2rad([-95, -10, -180]), np.deg2rad([-75, 10, 180]))
            sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
            target_goal[-1] += 0.3
            q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
            inv_done = False
            sample_times = 0
            while inv_done is False:
                if q_inv.max() > np.pi or q_inv.min() < -np.pi:
                    sim_euler = np.random.uniform(np.deg2rad([-95, 0, 0]), np.deg2rad([-85, 10, 10]))
                    sim_quat = Rotation.from_euler('xyz', sim_euler, degrees=False).as_quat()  # rotation convertor
                    q_inv = self.sim.inverse_kinematics_kdl(self.init_qpos, target_goal - base + ee_noise, sim_quat)
                    # print('change')
                    sample_times += 1
                    if sample_times > 500:
                        _reset_goal = True
                        # print(target_goal)
                        print('break')
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
        self.sim.set_forward()
        return _reset_goal

