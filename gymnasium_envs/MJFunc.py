import time
import mujoco
import mujoco.viewer
import numpy as np
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
import os

# import gymnasium_envs.KDLFunc as KDL_func
from ur_ikfast import ur_kinematics

from scipy.spatial.transform import Rotation

local_path = os.path.abspath('.') # get the excuted path (root path)

# rewrite xml
try:
    import xml.etree.cElementTree as ET
except:
    import xml.etree.ElementTree as ET


class MJFunc:
    def __init__(self,
                 render: bool = True,
                 xml_path: str = '',
                 xml_file_name: str = '',
                 ) -> None:
        self.root_path = local_path + xml_path
        self.xml_file = self.root_path + xml_file_name
        self.model = mujoco.MjModel.from_xml_path(self.xml_file)
        self.data = mujoco.MjData(self.model)
        self.render = render
        if self.render:
            # self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data) # package mujoco_viewer
            self.viewer_distance = 1.5  # set the sight posture
            self.viewer_azimuth = 270
            self.viewer_elevation = -45
            self.viewer_lookat = np.array([0, 1.3, 1.9])
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False) # raw mujoco viewer
            self.viewer.cam.distance = self.viewer_distance
            self.viewer.cam.azimuth = self.viewer_azimuth
            self.viewer.cam.elevation = self.viewer_elevation
            self.viewer.cam.lookat[:] = self.viewer_lookat
        # while True: # for testing mujoco render in python
        #     if self.render is True:
        #         mujoco.mj_step(self.model, self.data)
        #         # self.viewer.render()
        #         # self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(self.data.time % 2)
        #         self.viewer.sync()

        # hard code for urdf-kdl, can be embodied into yaml file
        kdl_urdf_file = self.root_path + 'ur5e_schunk.urdf'
        # self.kdl_solver = KDL_func.arm_kdl(kdl_urdf_file)
        # hard code for ikfast
        self.ur5e_arm = ur_kinematics.URKinematics('ur5e')


    def reload_xml(self, xml_file_name):
        if self.render:
            self.viewer.close()
        new_xml = self.root_path + xml_file_name
        self.model = mujoco.MjModel.from_xml_path(new_xml)
        self.data = mujoco.MjData(self.model)
        if self.render:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=False, show_right_ui=False)
            self.viewer.cam.distance = self.viewer_distance
            self.viewer.cam.azimuth = self.viewer_azimuth
            self.viewer.cam.elevation = self.viewer_elevation
            self.viewer.cam.lookat[:] = self.viewer_lookat
    @property
    def dt(self):
        '''timestep in Mjoaco, set in config file now, use set_time_step func'''
        pass

    def set_timestep(self, dt):
        self.model.opt.timestep = dt

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

    def step(self):
        mujoco.mj_step(self.model, self.data)
        if self.render is True:
            self.viewer.sync()

    def get_body_position(self, body: str) -> np.ndarray:
        position = self.data.xpos[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(position)

    def get_body_quaternion(self, body: str) -> np.ndarray:
        quat = self.data.xquat[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(quat)

    def get_body_euler(self, body: str, euler_dire='xyz') -> np.ndarray:
        quat = self.data.xquat[mujoco.mj_name2id(self.model, type=1, name=body)]
        roll_quat = np.roll(quat, -1)
        euler = Rotation.from_quat(roll_quat).as_euler(euler_dire, degrees=False)
        return euler

    def get_body_velocity(self, body: str) -> np.ndarray:
        vel = self.data.cvel[mujoco.mj_name2id(self.model, type=1, name=body)]
        return np.array(vel)

    def get_joint_angle(self, joint: str) -> float:
        return self.data.qpos[mujoco.mj_name2id(self.model, type=3, name=joint)]

    def get_joint_velocity(self, joint: str) -> float:
        return self.data.qvel[mujoco.mj_name2id(self.model, type=3, name=joint)]

    def get_site_position(self, site: str) -> np.ndarray:
        return self.data.site_xpos[mujoco.mj_name2id(self.model, type=6, name=site)]

    def get_site_quaternion(self, site: str) -> np.ndarray:
        mat = self.data.site_xmat[mujoco.mj_name2id(self.model, type=6, name=site)]
        quat = Rotation.from_matrix(mat.reshape((3, 3))).as_quat()
        return quat

    def get_site_euler(self, site: str, rot_type='xyz') -> np.ndarray:
        mat = self.data.site_xmat[mujoco.mj_name2id(self.model, type=6, name=site)]
        rot = Rotation.from_matrix(mat.reshape((3, 3))).as_euler(rot_type, degrees=False)
        return rot

    def get_site_mat(self, site: str) -> np.ndarray:
        return self.data.site_xmat[mujoco.mj_name2id(self.model, type=6, name=site)]

    def set_joint_qpos(self, joint_list, angles):
        assert len(joint_list) == len(angles)
        for i in range(len(joint_list)):
            self.data.joint(joint_list[i]).qpos = angles[i]
        mujoco.mj_forward(self.model, self.data)


    def set_mocap_pos(self, mocap: str, pos: np.ndarray) -> None:
        mocap_id = self.model.body_mocapid[self.data.body(mocap).id]
        self.data.mocap_pos[mocap_id] = pos
        # self.data.mocap_pos[mujoco.mj_name2id()]

    def set_mocap_quat(self, mocap: str, quat: np.ndarray) -> None:
        mocap_id = self.model.body_mocapid[self.data.body(mocap).id]
        self.data.mocap_quat[mocap_id] = quat

    def control_joints(self, actuator_list, target_angles) -> None:
        assert len(actuator_list) == len(target_angles)
        ids = []
        for i in range(len(actuator_list)):
            ids.append(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_list[i]))
        for i in range(len(target_angles)):
            self.data.ctrl[ids[i]] = target_angles[i]

    def set_forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    # def get_touch_sensor(self, sensor: str) -> float:
    #     return self.data.sensor(sensor)

    def get_ft_sensor(self, force_site: str, torque_site: str) -> np.ndarray:
        force = self.data.sensor(force_site).data
        torque = self.data.sensor(torque_site).data
        # mujoco.mj_contactForce()
        return np.hstack((force, torque))

    def _get_ft_sensor(self):
        for id, c in enumerate(self.data.contact):
            # ft = mujoco.mj_contactForce(self.model, self.data, id, ft)
            print(id)

    # def inverse_kinematics(self, current_joint: np.ndarray, target_position: np.ndarray,
    #                        target_orientation: np.ndarray) -> np.ndarray:
    #     qpos = self.urxkdl.inverse(current_joint, target_position, target_orientation)
    #     return qpos
    #
    # def forward_kinematics(self, qpos) -> np.ndarray:
    #     ee_pos = self.urxkdl.forward(qpos=qpos)
    #     return ee_pos

    # def inverse_kinematics_kdl(self,
    #                            current_joint: np.ndarray,
    #                            target_position: np.ndarray,
    #                            target_orientation: np.ndarray) -> np.ndarray:
    #     qpos = self.kdl_solver.inverse(current_joint, target_position, target_orientation)
    #     return qpos

    # def forward_kinematics_kdl(self, qpos) -> np.ndarray:
    #     ee_pos, ee_quat = self.kdl_solver.forward(qpos=qpos)
    #     return ee_pos, ee_quat

    def inverse_kinematics_ikfast(self, target_position, target_orientation, q_guess=np.zeros(6)):
        pose_quat = np.concatenate([target_position, target_orientation]) # xyz, rx, ry ,rz, w
        q_inv = self.ur5e_arm.inverse(pose_quat, q_guess=q_guess) # xyz, rx, ry ,rz, w
        return q_inv

    def forward_kinematics_ikfast(self, qpos):
        pose_quat = self.ur5e_arm.forward(qpos)
        return pose_quat[:3], pose_quat[3:] # xyz,  rx, ry ,rz, w

    def add_visual_capsule(self, scene, point1, point2, radius, rgba):
        """Adds one capsule to an mjvScene."""
        # TODO: useless now, trying to render new geom in mujoco
        if scene.ngeom >= scene.maxgeom:
            return
        scene.ngeom += 1  # increment ngeom
        # initialise a new capsule, add it to the scene using mjv_connector
        mujoco.mjv_initGeom(scene.geoms[scene.ngeom - 1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                            np.zeros(3), np.zeros(9), rgba.astype(np.float32))
        mujoco.mjv_connector(scene.geoms[scene.ngeom - 1],
                             mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                             point1, point2)

    def modify_scene(self, traj):
        """Draw position trace, speed modifies width and colors."""
        # TODO: useless now, trying to render new geom in mujoco
        scn = mujoco.Renderer(self.model).scene
        traj_len = traj.shape[1]
        sample_rate = 100
        sample_index = np.linspace(0, traj_len, int(traj_len/sample_rate), dtype=int)
        if len(sample_index) > 1:
            for i in range(len(sample_index) - 1):
                rgba = np.array((np.clip(0.5, 0, 1),
                                 np.clip(0.5, 0, 1),
                                 .5, 1.))
                radius = .003
                point1 = traj[:, sample_index[i]]

                point2 = traj[:, sample_index[i] + 1]
                print(point2)
                self.add_visual_capsule(scn, point1, point2, radius, rgba)
                # mujoco.Renderer(self.model).update_scene(self.data)
        # mujoco.Renderer(self.model).update_scene(self.data)
        # print(traj[:, 0])


    @contextmanager
    def no_rendering(self) -> Iterator[None]:
        pass  #


