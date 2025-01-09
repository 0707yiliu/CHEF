import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from scipy.spatial.distance import pdist
from typing import Callable


def quat_dis(q1, q2):
    '''
    distance between quaternions
    the normal calculation method for angle distance:
        q1 *. delta_q = q2
        delta_q = inv(q1) *. q2
                = (q1_w - q1_xi - q1_yj - q1_zk)(q2_w + q2_xi + q2_yj + q2_zk)
        delta_q' = |delta_q|
        delta_q' = cos(theta/2) + u*sin(theta/2) = delta_q'_w + delta_q'_xi + delta_q'yj + delta_q'_zk
        cos(theta/2) = delta_q'_w
        theta = 2 * arccos(delta_q'_w)
    the similarity:
        h = (q1w * q2w + q1x * q2x + q1y * q2y + q1z * q2z) / norm(q1) * norm(q2)
        h belong to [-1, 1]
        h = abs(h)

    '''
    # q1 = q1 / np.linalg.norm(q1)
    # q2 = q2 / np.linalg.norm(q2)
    # dis = 2 * np.arccos(abs(sum(q1 * q2)))
    # the similarity method
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    simularity = 1 - abs(sum(q1 * q2))
    return simularity # 0 -> 1, high -> low


def cus_log(x_value, base_x):
    """
    :param x_value: x
    :param base_x: base
    """
    return np.log(x_value) / np.log(base_x)


def reward_rescaling(loop_rew_datas, gamma=0.99):
    '''calculate the standard variation value for reward rescaling'''
    data_len = len(loop_rew_datas)
    discount_data = []
    for i in range(data_len):
        discount_data.append(loop_rew_datas[i] * (gamma ** ((data_len - 1) - i)))
    # return np.std(loop_rew_datas)
    return np.mean(discount_data), np.std(discount_data)


def linear_schedule(initial_value: float, lowest_value: float = 0.000, up=False) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float, up=up) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        if up is True:
            return (1 - progress_remaining) * initial_value + lowest_value
        else:
            return progress_remaining * initial_value + lowest_value

    return func

def lowpass_filter(last, cur, ratio=0.7):
    new = ratio * last + (1 - ratio) * cur
    return new

def circle_sample(center_x, center_y, diameter_in, diameter_out, thickness_low, thicness_high):
    """
    create the circle shape range for target point sampling for robot
    Args:
        center: the center of the circle, depended on the base pos of the robot
        diameter_in: the inside diameter of the circle
        diameter_out: the outside diameter of the circle
        thickness: the thickness in Z direction

    Returns:
        3-dim sample point
    """
    # generate diameter and circle shape like [dia, circle-shape] for x and y
    x_y_dia_circle = np.random.uniform([0.35 * np.pi, diameter_in], [0.65 * np.pi, diameter_out])

    x = center_x + x_y_dia_circle[1] * np.cos(x_y_dia_circle[0])
    y = center_y + x_y_dia_circle[1] * np.sin(x_y_dia_circle[0])
    z = np.random.uniform(thickness_low, thicness_high)

    return np.array([x, y, z])


def _normalization(data, _max, _min, range_max=1, range_min=0):
    if type(data) is not type(np.array([])):
        data = np.array(data)
    if type(_max) is not type(np.array([])):
        _max = np.array(_max)
        _min = np.array(_min)
    _range = _max - _min
    resize_range = range_max - range_min
    new_data = (resize_range * (data - _min) / _range) + range_min
    return new_data


def _inv_normalization(data, _range, _min, range_max=1, range_min=0):
    if type(data) is not type(np.array([])):
        data = np.array(data)
    if type(_range) is not type(np.array([])):
        _range = np.array(_range)
    if type(_min) is not type(np.array([])):
        _min = np.array(_min)
    resize_range = range_max - range_min
    raw_data = ((data - range_min) * _range / resize_range) + _min
    return raw_data


def euclidean_distance(a, b):
    assert len(a) == len(b)
    return np.linalg.norm(a - b)

def euler_angle_distance(a, b):
    '''the euler angle distance'''
    assert len(a) == len(b)
    a = np.cos(a)
    b = np.cos(b)
    dis = euclidean_distance(a, b)
    return dis

def cosine_distance(a, b):
    '''get the cosine distance of two vectors, not for euler vector'''
    # print(a.shape, b.shape)
    assert len(a) == len(b)
    dis = pdist(np.vstack([a, b]), 'cosine')[0]
    return dis


def load_single_discrete_pkl(root_path, pkl_root_name):
    """
    load trajectory/data from pkl, one pkl has one data
    !!! pre-processing func, not online
    Args:
        root_path: all .pkl's root path
        pkl_root_name: one .pkls folder file name

    Returns:
        rearranged data

    """
    import os
    demos = os.listdir(root_path)
    demo = pkl_root_name
    pickles = os.listdir(os.path.join(root_path, demo))
    pickles.sort()
    positions = []
    import pickle
    import matplotlib.pyplot as plt
    qpos = []
    ee_pos = []
    ee_rot = []
    force_torque = []
    for pickle_path in pickles:
        # dict_keys(['joint_positions', 'tcp_pose_rotvec', 'wrench', 'gripper_position', 'control'])
        # mainly use joint_positions, tcp_pose_rotvec and wrench (F/T sensor)
        obs = pickle.load(open(os.path.join(root_path, demo, pickle_path), "rb"))
        joint_position = obs['joint_positions']
        tcp_pose = obs['tcp_pose_rotvec'][:3]
        tcp_rot = obs['tcp_pose_rotvec'][3:]
        ft = obs["wrench"]
        qpos.append(joint_position)
        ee_pos.append(tcp_pose)
        ee_rot.append(tcp_rot)
        force_torque.append(ft)
    qpos = np.array(qpos)
    ee_pos = np.array(ee_pos)
    ee_rot = np.array(ee_rot)
    force_torque = np.array(force_torque)
    # print(ee_pos.shape)

    import gymnasium_envs.KDLFunc as KDL_func
    kdl_urdf_file = './robot_env_description/' + 'ur5e_schunk.urdf'
    # the mjx has been matched to this urdf with real robot
    kdl_solver = KDL_func.arm_kdl(kdl_urdf_file)
    # ee_pos_kdl, ee_qua_kdl = kdl_solver.forward(qpos=qpos[3, :-1])
    ee_pos_kdl = np.empty([0, 3])
    ee_quat_kdl = np.empty([0, 4])
    for i in range(qpos.shape[0]):
        eepos, eequat = kdl_solver.forward(qpos=qpos[i, :-1])
        ee_pos_kdl = np.append(ee_pos_kdl, [eepos], axis=0)
        ee_quat_kdl = np.append(ee_quat_kdl, [eequat], axis=0)
    print('ee pos length:', len(ee_pos_kdl))
    start = 0
    end = -100
    ee_pos_kdl = ee_pos_kdl[start:end, :] # TODO !!!!! important to divide data for trajectory
    ee_quat_kdl = ee_quat_kdl[start:end, :]
    force_torque = force_torque[start:end, :]
    qpos = qpos[start:end, :]

    # ee_pos_kdl = np.flipud(ee_pos_kdl)
    # ee_quat_kdl = np.flipud(ee_quat_kdl)
    # force_torque = np.flipud(force_torque)
    # qpos = np.flipud(qpos)

    # import matplotlib.pyplot as plt
    #
    # # -------- EEF tcp pos rotvec plot -----
    # import open3d
    # import cv2
    # pcd = open3d.geometry.PointCloud()
    # pcd.points = open3d.utility.Vector3dVector(ee_pos_kdl)
    # colors = []
    # for i in range(len(ee_pos_kdl)):
    #     color = cv2.applyColorMap(np.array(i / len(ee_pos_kdl) * 255, dtype=np.uint8), cv2.COLORMAP_CIVIDIS)
    #     colors.append(color[0][0] / 255)
    #
    # pcd.colors = open3d.utility.Vector3dVector(colors)
    #
    # # add a coordinate frame of the robot
    # frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    #
    # open3d.visualization.draw_geometries([pcd, frame])
    # # ----------------------------------------
    # input()
    import mujoco
    import mujoco.viewer
    import time
    root_path = 'robot_env_description/scene_pour.xml'
    model = mujoco.MjModel.from_xml_path(root_path)
    data = mujoco.MjData(model)
    viewer_distance = 1.5  # set the sight posture
    viewer_azimuth = 270
    viewer_elevation = -45
    viewer_lookat = np.array([0, 1.3, 1.9])
    viewer = mujoco.viewer.launch_passive(model, data, show_left_ui=False,
                                               show_right_ui=False)  # raw mujoco viewer
    viewer.cam.distance = viewer_distance
    viewer.cam.azimuth = viewer_azimuth
    viewer.cam.elevation = viewer_elevation
    viewer.cam.lookat[:] = viewer_lookat
    joint_list = ["shoulder_pan_jointL", "shoulder_lift_jointL", "elbow_jointL", "wrist_1_jointL", "wrist_2_jointL", "wrist_3_jointL", "finger_joint1"]
    for i in range(6):
        data.joint(joint_list[i]).qpos = qpos[0, i]
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data)
    viewer.sync()
    time.sleep(0.1)
    for j in range(ee_pos_kdl.shape[0]):
        angles = qpos[j, :-1]
        # print(angles)
        for i in range(len(joint_list[:-1])):
            data.ctrl[i] = angles[i]
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.01)
    print('done')
    ## ---------record processed data -----------
    input()
    '''
    saving processed data
    '''
    import datetime
    import time
    currenttime = int(time.time())
    currenttime = time.strftime("%Y%m%d%H%M%S", time.localtime(currenttime))
    print(currenttime)
    np.savez('./datasets/pour/' + currenttime,
             eefpos=ee_pos_kdl,
             eefquat=ee_quat_kdl,
             eefft=force_torque)

    # # -------- F/T sensor plot -----
    # ft_wrench = []
    # for pickle_path in pickles:
    #     obs = pickle.load(open(os.path.join(root_path, demo, pickle_path), "rb"))
    #     ft = obs["wrench"]
    #     ft_wrench.append(ft)
    #
    # ft_wrench = np.array(ft_wrench)
    #
    # # plot the timeseries
    # plt.plot(ft_wrench)
    # plt.xlabel("time")
    # plt.ylabel("force (N) / torque (Nm)")
    # plt.legend(["fx", "fy", "fz", "tx", "ty", "tz"])
    # plt.show()
    # # --------------------------------

def interp_preprocessed_data_with_vel(data_path, ex_length=2000, hz=50):
    """
    data extender and generate velocity
    Args:
        data_path: datasets path

    Returns:
        interped data

    """
    data = np.load(data_path)
    # print('load_preprocesseddata_with_vel func:', data.files, data['eefpos'].shape)
    eepos = data['eefpos']
    eeeuler = Rotation.from_quat(data['eefquat']).as_euler('xyz', degrees=False)
    forcetorque = data['eefft']
    # plt.subplot(131)
    # plt.plot(eepos[:, 0])
    # plt.plot(eeeuler[:, 0])
    # plt.subplot(132)
    # plt.plot(eepos[:, 1])
    # plt.plot(eeeuler[:, 1])
    # plt.subplot(133)
    # plt.plot(eepos[:, 2])
    # plt.plot(eeeuler[:, 2])
    # plt.show()
    hz = hz
    extender_length = ex_length
    raw_len = eepos.shape[0]
    ratio = extender_length / raw_len
    pos_vel = np.empty([0, 3])
    rot_vel = np.empty([0, 3])
    for i in range(data['eefpos'].shape[0]-1):
        _pos_vel = (eepos[i+1] - eepos[i]) * hz
        _rot_vel = (eeeuler[i + 1] - eeeuler[i]) * hz
        pos_vel = np.append(pos_vel, [_pos_vel], axis=0)
        rot_vel = np.append(rot_vel, [_rot_vel], axis=0)
    pos_vel = np.append(pos_vel, [[0, 0 ,0]], axis=0) # the end is zero
    rot_vel = np.append(rot_vel, [[0, 0, 0]], axis=0) # the end is zero

    raw_data_len = eepos.shape[0]
    raw_t = np.linspace(0, raw_data_len-1, raw_data_len)
    ex_t = np.linspace(0, raw_data_len-1, extender_length)
    ex_eepos = np.zeros((extender_length, 3))
    ex_eeposvel = np.zeros_like(ex_eepos)
    ex_eerot = np.zeros_like(ex_eepos)
    # ex_eequat = np.zeros((extender_length, 4)) # quaternion
    ex_eerotvel = np.zeros_like(ex_eepos)
    ex_forcetorque = np.zeros((extender_length, 6))
    for i in range(3):
        ex_eepos[:, i] = np.interp(ex_t, raw_t, eepos[:, i])
        ex_eerot[:, i] = np.interp(ex_t, raw_t, eeeuler[:, i])
        ex_eeposvel[:, i] = np.interp(ex_t, raw_t, pos_vel[:, i]) / ratio
        ex_eerotvel[:, i] = np.interp(ex_t, raw_t, rot_vel[:, i]) / ratio
    for i in range(6):
        ex_forcetorque[:, i] = np.interp(ex_t, raw_t, forcetorque[:, i])
    ex_eequat = Rotation.from_euler('xyz', ex_eerot, degrees=False).as_quat()

    # plt.subplot(131)
    # plt.plot(pos_vel[:, 0])
    # plt.plot(rot_vel[:, 0])
    # plt.subplot(132)
    # plt.plot(pos_vel[:, 1])
    # plt.plot(rot_vel[:, 1])
    # plt.subplot(133)
    # plt.plot(pos_vel[:, 2])
    # plt.plot(rot_vel[:, 2])
    # plt.show()

    return ex_eepos.T, ex_eerot.T, ex_eequat.T, ex_eeposvel.T, ex_eerotvel.T, ex_forcetorque.T
    # transform to normal shape like (3, ex_length)




if __name__ == "__main__":
    print('main')
    # ## pre process the demostration trajectory --------
    # root_path = "/home/yi/robotic_manipulation/demonstration_datasets/chef_skills/pour/"
    # file_name = "1203_122009" # get the data one by one
    #
    # load_single_discrete_pkl(root_path, file_name)
    # # ---------------------

    ## --------------precessed data load and get vel and interp ----------
    path = './datasets/reach/'
    filename = '20241204105546.npz'
    interp_preprocessed_data_with_vel(path + filename)
    ## -------------------------

