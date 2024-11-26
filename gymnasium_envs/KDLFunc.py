import math
import numpy as np
import PyKDL as kdl

from kdl_parser.urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_parser import kdl_tree_from_urdf_model
from scipy.spatial.transform import Rotation as R

class arm_kdl():
    def __init__(self, DHfile) -> None:
        robot = URDF.from_xml_file(DHfile)
        tree = kdl_tree_from_urdf_model(robot)
        self.chain = tree.getChain("base_link", "tool0")  # hard code in URDF-file

    def forward(self, qpos):
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        pos = kdl.Frame()
        q = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            q[i] = qpos[i]
        fk_flag = fk.JntToCart(q, pos)
        f_pos = np.zeros(3)
        for i in range(3):
            f_pos[i] = pos.p[i]
        return f_pos

    def inverse(self, init_joint, goal_pose, goal_rot):
        try:
            rot = kdl.Rotation()
            rot = rot.Quaternion(goal_rot[0], goal_rot[1], goal_rot[2], goal_rot[3])  # radium x y z w
            pos = kdl.Vector(goal_pose[0], goal_pose[1], goal_pose[2])
        except ValueError:
            print("The target pos can not be transfor to IK-function.")
        target_pos = kdl.Frame(rot, pos)
        # print(target_pos)
        fk = kdl.ChainFkSolverPos_recursive(self.chain)
        # inverse kinematics
        ik_v = kdl.ChainIkSolverVel_pinv(self.chain)

        ik_p_kdl = kdl.ChainIkSolverPos_NR(self.chain, fk, ik_v)
        q_init = kdl.JntArray(self.chain.getNrOfJoints())
        for i in range(6):
            q_init[i] = init_joint[i]
        q_out = kdl.JntArray(self.chain.getNrOfJoints())
        ik_p_kdl.CartToJnt(q_init, target_pos, q_out)
        # print("Output angles:", q_out)
        q_out_trans = np.zeros(self.chain.getNrOfJoints())
        for i in range(self.chain.getNrOfJoints()):
            q_out_trans[i] = np.array(q_out[i])
        # print(q_out_trans)
        return (q_out_trans)