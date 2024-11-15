import time
import mujoco
import mujoco_viewer
import numpy as np
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
import os

local_path = os.path.abspath('.') # get the run.py path (root path)

class MJFunc:
    def __init__(self,
                 render: bool = True,
                 xml_path: str = '/gymnasium_envs/dualarm_chef_description/scene.xml',
                 ) -> None:
        self.xml_file = local_path + xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.render = render
        if self.render:
            self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)

    def reload_xml(self):
        pass

    @property
    def dt(self):
        pass

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.render is True:
            self.viewer.render()

    def get_body_position(self, body: str) -> np.ndarray:
        position = self.data.xpos[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(position)

    def get_body_quaternion(self, body: str) -> np.ndarray:
        quat = self.data.xquat[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(quat)

    def get_body_velocity(self, body: str) -> np.ndarray:
        vel = self.data.cvel[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(vel)

    def get_joint_angle(self, joint: str) -> float:
        return self.data.qpos[mujoco.mj_name2id(self.model, type=3, name=joint)]

    def get_joint_velocity(self, joint: str) -> float:
        return self.data.qvel[mujoco.mj_name2id(self.model, type=3, name=joint)]

    def get_site_position(self, site: str) -> np.ndarray:
        return self.data.site_xpos[mujoco.mj_name2id(self.model, type=6, name=site)]

    def get_site_mat(self, site: str) -> np.ndarray:
        return self.data.site_xmat[mujoco.mj_name2id(self.model, type=6, name=site)]

    def set_joint_angles(self, angles: np.ndarray) -> None:
        for i in range(len(angles)):
            self.data.qpos[i] = angles[i]
        mujoco.mj_forward(self.model, self.data)

    def set_mocap_pos(self, mocap: str, pos: np.ndarray) -> None:
        self.data.mocap_pos[0] = pos  # TODO:the id is not defined in mujoco, you should design a search method
        # self.data.mocap_pos[mujoco.mj_name2id()]

    def set_mocap_quat(self, mocap: str, quat: np.ndarray) -> None:
        self.data.mocap_quat[0] = quat  # TODO: the same problem like set_mocap_pos func

    def control_joints(self, target_angles: np.ndarray) -> None:
        for i in range(len(target_angles)):
            self.data.ctrl[i] = target_angles[i]

    def set_forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # def get_touch_sensor(self, sensor: str) -> float:
    #     return self.data.sensor(sensor)

    def get_ft_sensor(self, force_site: str, torque_site: str) -> np.ndarray:
        force = self.data.sensor(force_site).data
        torque = self.data.sensor(torque_site).data
        return np.hstack((force, torque))

    # def inverse_kinematics(self, current_joint: np.ndarray, target_position: np.ndarray,
    #                        target_orientation: np.ndarray) -> np.ndarray:
    #     qpos = self.urxkdl.inverse(current_joint, target_position, target_orientation)
    #     return qpos
    #
    # def forward_kinematics(self, qpos) -> np.ndarray:
    #     ee_pos = self.urxkdl.forward(qpos=qpos)
    #     return ee_pos

    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        pass  #


