import sys
import rtde_receive
import rtde_control
import time

from matplotlib import pyplot as plt
import numpy as np

from admittance_controller.core import FT_controller as AdmController
from scipy.spatial.transform import Rotation as R

from kdl_func import KDL_ROBOT

sample_time = []
force_xyz = []
force_rxyz = []
i = 0

rtde_c = rtde_control.RTDEControlInterface("10.42.0.162")
rtde_r = rtde_receive.RTDEReceiveInterface("10.42.0.162")

# ------------- admittance controller -------------
rtde_c.zeroFtSensor()
# print('zero force:', rtde_r.getActualTCPForce())
force_controller = AdmController(0.5, 1000, 5, 0.01)
admittance_params = np.zeros((3, 3)) # contains acc, vel and pos in xyz derictions
admittance_paramsT = np.zeros((3, 3))
# -------------------------------------------------
# ------- KDL ------------
ur3e_kdl = KDL_ROBOT('/home/yi/robotic_manipulation/SoftBodySlicing/ur3e.urdf', "base_link", "tool0")
# ----------------------

# joint_angles_curr = rtde_r.getActualQ()
# # print(joint_angles_curr)
# match_angle = np.array([0, 3.14, -0.000550976546946913, -0.01395626485858159, -0.007717911397115529, 0.006361254956573248])
# match_angle = rtde_r.getActualQ()
# # ------------- KDL test ---------------
# f_pos, kdl_f_qua = ur3e_kdl.forward(match_angle)
# print('KDL pos and qua:', f_pos, kdl_f_qua)
# print('ee pos', rtde_r.getActualTCPPose()[:3])
# r_target_rot = R.from_rotvec(rtde_r.getActualTCPPose()[3:], degrees=False)
# r_target_qua = r_target_rot.as_quat()
# print('ee qua:', r_target_qua)
# print('ee q:', joint_angles_curr)
# # f_pos[2] -= 0.01
# ik_q = ur3e_kdl.inverse(rtde_r.getActualQ(), f_pos, kdl_f_qua)
# print('ik q:', ik_q)
# # rtde_c.servoJ(ik_q, 0.001, 0.001, 0.5, 0.03, 500)
# # ----------------------
# # testq = np.array([-0.18713871, -1.54668583,  2.21557594, -1.78615128, -0.38870288,  0.43094718])
# # f_pos, kdl_f_qua = ur3e_kdl.forward(testq)
# # print('test:', f_pos, kdl_f_qua)
# -----------------------------------------
ratio = 0.7
last_ft = np.zeros(6)
last_pos = np.array(rtde_r.getActualTCPPose())
def lowpass_filter(last, cur, ratio):
    new = ratio * last + (1 - ratio) * cur
    return new
match_angle = rtde_r.getActualQ()
des_pos, kdl_f_qua = ur3e_kdl.forward(match_angle)
des_qua = R.from_quat(kdl_f_qua)
des_rot = R.from_rotvec(rtde_r.getActualTCPPose()[3:], degrees=False)
des_euler = des_qua.as_euler('xyz', degrees=False)
truncation_num = 4
des_pos = np.around(des_pos, truncation_num) # truncation for float number
des_euler = np.around(des_euler, truncation_num)
adm_control_actived = False
ft_rot_th = 0.5
ft_pos_th = 10
while True:
    # TODO: data cutting for ik
    time.sleep(0.01)
    # curr_ft = np.array(rtde_r.getActualTCPForce())
    curr_ft = np.around(rtde_r.getActualTCPForce(), truncation_num)
    # curr_pos = np.array(rtde_r.getActualTCPPose())[:3]
    # curr_rot = R.from_rotvec(rtde_r.getActualTCPPose()[3:], degrees=False)
    # curr_euler = r_target_rot.as_euler('xyz', degrees=False)
    # print(last_ft, curr_ft)
    last_ft = np.around(lowpass_filter(last_ft, curr_ft, ratio), truncation_num)

    position_d, rotation_d, admittance_params, admittance_paramsT = force_controller.admittance_control(
                                              desired_position=des_pos,
                                              desired_rotation=des_euler,
                                              FT_data=last_ft,
                                              params_mat=admittance_params,
                                              paramsT_mat=admittance_paramsT)
    position_d = np.around(position_d, truncation_num)
    rotation_d = np.around(rotation_d, truncation_num)
    r_target_rot = R.from_euler('xyz', rotation_d, degrees=False)
    r_target_qua = r_target_rot.as_quat()
    ik_q = ur3e_kdl.inverse(rtde_r.getActualQ(), position_d, r_target_qua)
    ik_q = np.around(ik_q, truncation_num)
    # print(ik_q)
    # --------- zero force control (separate as one func)------------
    # if np.any(last_ft[3:] > ft_rot_th) or np.any(last_ft[3:] < -ft_rot_th) or np.any(last_ft[:3] > ft_pos_th) or np.any(last_ft[:3] < -ft_pos_th):
    #     # for recording the desired posture but not set to des_pos and des_euler
    #     adm_control_actived = True
    #     temp_des_pos = position_d
    #     temp_des_euler = rotation_d
    # if np.all(last_ft[3:] < ft_rot_th) and np.all(last_ft[3:] > -ft_rot_th) and np.all(last_ft[:3] < ft_pos_th) and np.all(last_ft[:3] > -ft_pos_th) and adm_control_actived is True:
    #     # zero force control for setting des_pos and des_euler when no ext-force
    #     # print(des_pos)
    #     zero_force_actived = True
    #     # deactivate adm control
    #     adm_control_actived = False
    #     admittance_params = np.zeros((3, 3))  # contains acc, vel and pos in xyz derictions
    #     admittance_paramsT = np.zeros((3, 3))
    #     des_pos = temp_des_pos
    #     des_euler = temp_des_euler
    # ---------------------------------------------
    print('r qua:', r_target_qua)
    print('ee q:', rtde_r.getActualQ())
    print('ik q:', ik_q)
    rtde_c.servoJ(ik_q, 0.001, 0.001, 0.5, 0.03, 500) # moving func !!!!!!!!!!!!!!!!!!!!!!!!
    try:
        i += 1
        sample_time.append(i)
        # force_xyz.append(np.hstack((position_d, rotation_d)))
        # force_xyz.append(position_d)
        force_xyz.append(last_ft[:3])
        plt.clf()  # 清除之前画的图
        plt.plot(sample_time, force_xyz)  # 画出当前x列表和y列表中的值的图形
        plt.pause(0.001)  # 暂停一段时间，不然画的太快会卡住显示不出来
        plt.ioff()  # 关闭画图窗口

    except KeyboardInterrupt:
        time.sleep(1)
        print('over')
        plt.clf()
        plt.close()
        sys.exit(0)
