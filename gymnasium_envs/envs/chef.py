from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from gymnasium_envs.envs.core import RobotTaskEnv
from gymnasium_envs.envs.robots.dual_UR5e import dualUR5e
from gymnasium_envs.envs.tasks.kitchen import KitchenMultiTask
from gymnasium_envs.MJFunc import MJFunc

class ChefEnv_v0(RobotTaskEnv):
    def __init__(self,
                 render: bool = True,
                 ):

        sim = MJFunc(
            render=render,
        )

        robot = dualUR5e(
            sim=sim,
        )

        task = KitchenMultiTask(
            sim=sim
        )

        super().__init__(
            robot,
            task,
            render=render
        )

