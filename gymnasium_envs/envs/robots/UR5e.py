import numpy as np
import math
from gymnasium_envs.envs.core import MJRobot
from gymnasium import spaces

from gymnasium_envs.utils import circle_sample

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
        L_ee_qua = np.copy(self.sim.get_body_quaternion(self.L_eef))
        R_ee_pos = np.copy(self.sim.get_body_position(self.R_eef))
        R_ee_qua = np.copy(self.sim.get_body_quaternion(self.R_eef))
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        R_FT_sensor = self.sim.get_ft_sensor('Rforce', 'Rtorque')
        obs = np.concatenate([L_ee_pos, L_ee_qua, L_FT_sensor,
                              R_ee_pos, R_ee_qua, R_FT_sensor])

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

        super().__init__(
            sim,
            action_space=action_space,
            joint_index=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_force=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]),
            joint_list=["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"],
            sensor_list=['magnetic_1', 'magnetic_2', 'magnetic_3', 'magnetic_4']
        )


    def set_action(self, action: np.ndarray) -> None:
        self.sim.step()
        # ---KDL Test (work)---
        # test_angle_zero = np.zeros(6)
        # # test_angle = [0.32350003, 0.62122977, 0.9344792, 2.95638272, -0.12349997, -2.61809169]
        # test_angle = [0.3235, 0.7235,0.2235,0.4235,0.1235,0.5235]
        # self.sim.set_joint_qpos(self.joint_list[:-1], test_angle)
        # qpos = np.array(test_angle)
        # # print(np.array([self.sim.get_joint_angle(joint=self.joint_list[i]) for i in range(7)]))
        # if self.env_index == 1:
        #     base = self.sim.get_body_position('baseL')
        #     sim_pos = self.sim.get_body_position('grab_obj')
        #     sim_qua = np.roll(self.sim.get_body_quaternion('grab_obj'), -1)
        #     kdl_pos, kdl_qua = self.sim.forward_kinematics_kdl(qpos)
        #     q_inv = self.sim.inverse_kinematics_kdl(test_angle_zero, sim_pos - base,
        #                                              sim_qua)
        #     print('-------------', '\n',
        #         'sim pos:', sim_pos, '\n',
        #         'sim qua:', sim_qua, '\n',
        #         'kdl pos:', kdl_pos + base, '\n',
        #         'kdl qua:', kdl_qua, '\n',
        #         'kdl q inv:', q_inv,
        #     )
        # ---------------------


    def get_obs(self) -> np.ndarray:
        L_ee_pos = np.copy(self.sim.get_body_position(self.L_eef))
        L_ee_qua = np.copy(self.sim.get_body_quaternion(self.L_eef))
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        obs = np.concatenate([L_ee_pos, L_ee_qua, L_FT_sensor,])

        return obs

    def reset(self, index, task_result) -> None:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        self.env_index = index



        # hard code for different skill's environment
        # if index == 0:  # reach skill, init the robot to the fixed posture
        #     self.sim.set_joint_angles()
        # elif index == 1:  # flip skill, use IK to generate one posture, fix the Z-rotation, face to the ground
        #     goaled_qpos = self.sim.inverse_kinematics_kdl(current_joint=[0,1,2,3,4,5],
        #                                                   target_position=task_result,
        #                                                   target_orientation=task_result)
        #     # TODO: change the current_joint to the init joint, the qpos will be changed by ik,
        #     #       change the target orientation
        #     self.sim.set_joint_angles(goaled_qpos)
        #
        # elif index == 2:  # pouring cube, set the position of the end-effector upon the fixed area
        #     target_goal = np.copy(task_result)
        #     target_goal[-1] += 0.1
        #     goaled_qpos = self.sim.inverse_kinematics_kdl(current_joint=[0, 1, 2, 3, 4, 5],
        #                                                   target_position=task_result,
        #                                                   target_orientation=task_result)
        #     # TODO: change the current_joint to the init joint, the qpos will be changed by ik,
        #     #       change the target orientation
        #     self.sim.set_joint_angles(goaled_qpos)

        self.sim.set_forward()

