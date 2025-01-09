import matplotlib.pyplot as plt
import os
import numpy as np
import random
from gymnasium_envs.utils import interp_preprocessed_data_with_vel
from scipy.spatial.transform import Rotation
"""plot the 3d data for position and choose the reasonable data for training"""
skill = 'flip'
path = './'+skill+'/'
data_names = os.listdir(path)
print(data_names)
target_path = './clean_' + skill + '/'
data_path = path + random.choice(data_names)
for i in range(len(data_names)):
    demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
        data_path=path + data_names[i],
        ex_length=2000,
    )
    demo_ee_rot = Rotation.from_euler('xyz', demo_ee_rot.T).as_euler('zyx')
    demo_ee = np.vstack((demo_ee_pos, demo_ee_rot.T))
    fig = plt.figure(figsize=(10, 10))
    for plt_index in range(1, 7):
        ax = fig.add_subplot(2,3,plt_index)
        ax.plot(demo_ee[plt_index-1, :])
    plt.show()
#
# import open3d
# import cv2
# for i in range(len(data_names)):
#     demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
#         data_path=path + data_names[i],
#         ex_length=2000,
#     )
#     print(data_names[i])
#     demo_ee_pos_list = demo_ee_pos.T.tolist()
#     # print(demo_ee_pos_list)
#     pcd = open3d.geometry.PointCloud()
#     pcd.points = open3d.utility.Vector3dVector(demo_ee_pos_list)
#     colors = []
#     for i in range(len(demo_ee_pos_list)):
#         color = cv2.applyColorMap(np.array(i / len(demo_ee_pos_list) * 255, dtype=np.uint8), cv2.COLORMAP_CIVIDIS)
#         colors.append(color[0][0] / 255)
#
#     pcd.colors = open3d.utility.Vector3dVector(colors)
#
#     # add a coordinate frame of the robot
#     frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
#
#     open3d.visualization.draw_geometries([pcd, frame])