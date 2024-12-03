import time

import numpy as np


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
    x_y_dia_circle = np.random.uniform([0, diameter_in], [2 * np.pi, diameter_out])

    x = center_x + x_y_dia_circle[1] * np.cos(x_y_dia_circle[0])
    y = center_y + x_y_dia_circle[1] * np.sin(x_y_dia_circle[0])
    z = np.random.uniform(thickness_low, thicness_high)

    return np.array([x, y, z])


def _normalization(data, _max, _min):
    if type(data) is not type(np.array([])):
        data = np.array(data)
    if type(_max) is not type(np.array([])):
        _max = np.array(_max)
        _min = np.array(_min)
    _range = _max - _min
    return (data - _min) / _range

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)


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
    ee_pos_kdl = ee_pos_kdl[750:, :] # TODO !!!!! important to divide data for trajectory

    # import matplotlib.pyplot as plt
    #
    # # -------- EEF tcp pos rotvec plot -----
    import open3d
    import cv2
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(ee_pos_kdl)
    colors = []
    for i in range(len(ee_pos_kdl)):
        color = cv2.applyColorMap(np.array(i / len(ee_pos_kdl) * 255, dtype=np.uint8), cv2.COLORMAP_CIVIDIS)
        colors.append(color[0][0] / 255)

    pcd.colors = open3d.utility.Vector3dVector(colors)

    # add a coordinate frame of the robot
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    open3d.visualization.draw_geometries([pcd, frame])
    # # ----------------------------------------
    input()
    '''
    saving processed data
    '''
    import datetime
    import time
    currenttime = int(time.time())
    currenttime = time.strftime("%Y%m%d%H%M%S", time.localtime(currenttime))
    print(currenttime)
    np.savez('./datasets/reach/' + currenttime,
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




if __name__ == "__main__":
    root_path = "/home/yi/robotic_manipulation/demonstration_datasets/chef_skills/reach/"
    file_name = "1203_114756" # get the data one by one
    # 114756 is done ///!!!!!!!!!!!

    load_single_discrete_pkl(root_path, file_name)

