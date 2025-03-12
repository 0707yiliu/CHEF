import numpy as np
import rtde_receive
rtde_r = rtde_receive.RTDEReceiveInterface("10.42.0.162")
actual_q = rtde_r.getActualQ()
print('get q:', actual_q)

import rtde_control
import rtde_receive
import os
from ur_ikfast import ur_kinematics

from scipy.spatial.transform import Rotation as R

# rtde_c = rtde_control.RTDEControlInterface("10.42.0.162")

# Parameters
velocity = 0.5
acceleration = 0.5
dt = 1.0/500  # 2ms
lookahead_time = 0.1
gain = 300
joint_q = [-0.21761899 ,-1.83153582 ,-1.38600528, -0.94102257  ,1.38188696 ,-2.19524479]

# rtde_c.servoJ(joint_q, velocity, acceleration, dt, lookahead_time, gain)

local_path = os.path.abspath('.') # get the excuted path (root path)
xml_path = '/gymnasium_envs/robot_env_description/'
root_path = local_path + xml_path
kdl_urdf_file = root_path + 'ur5e_schunk.urdf'

ur5e_arm = ur_kinematics.URKinematics('ur5e')

pose_quat = ur5e_arm.forward(actual_q)
print('ikfast ee pos:', pose_quat)
ik_quat = pose_quat[3:]
# print(ik_quat)
ik_rot = R.from_quat(ik_quat, scalar_first=False).as_euler('zyx')
print('ikfast rot:', ik_rot, np.rad2deg(ik_rot))
rtde_eepos = rtde_r.getActualTCPPose()
print('rtde ee pos:', rtde_eepos)
print('rtde ee rot:', rtde_eepos[3:], np.rad2deg(rtde_eepos[3:]))
rtde_ee_quat = R.from_euler('xyz', rtde_eepos[3:]).as_quat(scalar_first=False)
print('rtde real quat:', rtde_ee_quat)
rtde_ee_rot_zyx = R.from_quat(rtde_ee_quat, scalar_first=False).as_euler('zyx')
print('rtde eerot zyx:', rtde_ee_rot_zyx, np.rad2deg(rtde_ee_rot_zyx))

# ee_pos_info = np.array([0.66936438,-0.33780115 , 0.36842845 , 0.94308969  ,0.16815638 , 0.21129314, 0.19406304]) # x y z xyzw
inv_q = ur5e_arm.inverse(pose_quat, q_guess=actual_q)
print('inv q:', inv_q)


