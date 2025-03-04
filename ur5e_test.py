import numpy as np
import rtde_receive
rtde_r = rtde_receive.RTDEReceiveInterface("10.42.0.162")
actual_q = rtde_r.getActualQ()
print('get q:', actual_q)

import rtde_control
import rtde_receive
import os
from ur_ikfast import ur_kinematics

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
rtde_eepos = rtde_r.getActualTCPPose()
print('rtde ee pos:', rtde_eepos)

ee_pos_info = np.array([ 0.66936438 ,-0.33780115 , 0.36842845 , 0.94308969  ,0.16815638 , 0.21129314,
  0.19406304]) # x y z xyzw
inv_q = ur5e_arm.inverse(ee_pos_info, q_guess=[2.4739255905151367, -1.306336687212326, 1.3867653051959437, -2.230870863000387, -1.5151923338519495, 3.7083895206451416])
print('inv q:', inv_q)


