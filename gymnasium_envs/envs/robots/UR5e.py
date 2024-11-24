import numpy as np
import math
from gymnasium_envs.envs.core import MJRobot
from gymnasium import spaces

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
        L_FT_sensor = self.sim.get_ft_sensor('Lforce', 'Ltorque')
        obs = np.concatenate([L_ee_pos, L_ee_qua, L_FT_sensor,])

        return obs

    def reset(self) -> None:
        """
        reset the simulator first and reset the robot posture then
        Returns:
            None
        """
        self.sim.reset()


