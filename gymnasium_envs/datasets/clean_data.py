import matplotlib.pyplot as plt
import os
import numpy as np
import random
from gymnasium_envs.utils import interp_preprocessed_data_with_vel
"""plot the 3d data for position and choose the reasonable data for training"""
skill = 'reach'
path = './'+skill+'/'
data_names = os.listdir(path)
print(data_names)
target_path = './clean_' + skill + '/'
# data_path = path + random.choice(data_names)
# for i in range(len(data_names)):
#     demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
#         data_path=path + data_names[i],
#         ex_length=2000,
#     )
#     fig = plt.figure()
#     ax = fig.add_subplot(projection = '3d')
#     ax.plot(demo_ee_pos[:, 0], demo_ee_pos[:, 1], demo_ee_pos[:, 2], label='parametric curve')
#     ax.legend()
#     ax.set_xlim(-1, 1)
#     ax.set_ylim(-1, 1)
#     ax.set_zlim(-1, 1)
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('z')
#     plt.show()

import open3d
import cv2
for i in range(len(data_names)):
    demo_ee_pos, demo_ee_rot, demo_ee_posvel, demo_ee_rotvel, demo_ee_quat, demo_eeft = interp_preprocessed_data_with_vel(
        data_path=path + data_names[i],
        ex_length=2000,
    )
    demo_ee_pos_list = demo_ee_pos.T.tolist()
    # print(demo_ee_pos_list)
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(demo_ee_pos_list)
    colors = []
    for i in range(len(demo_ee_pos_list)):
        color = cv2.applyColorMap(np.array(i / len(demo_ee_pos_list) * 255, dtype=np.uint8), cv2.COLORMAP_CIVIDIS)
        colors.append(color[0][0] / 255)

    pcd.colors = open3d.utility.Vector3dVector(colors)

    # add a coordinate frame of the robot
    frame = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

    open3d.visualization.draw_geometries([pcd, frame])