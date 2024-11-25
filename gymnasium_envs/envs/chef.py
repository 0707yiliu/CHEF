from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from gymnasium_envs.envs.core import RobotTaskEnv
from gymnasium_envs.envs.robots.UR5e import dualUR5e, singleUR5e
from gymnasium_envs.envs.tasks.kitchen import KitchenMultiTask
from gymnasium_envs.MJFunc import MJFunc


class ChefEnv_v0(RobotTaskEnv):
    def __init__(self,
                 render: bool = True,
                 xml_path: str = '',
                 xml_file_name: str = '',
                 basic_skills: list = [],
                 specified_skills: list = [],
                 ):

        sim = MJFunc(
            render=render,
            xml_path=xml_path,
            xml_file_name=xml_file_name,
        )

        robot = singleUR5e(
            sim=sim,
        )

        task = KitchenMultiTask(
            sim=sim,
            basic_skills=basic_skills, # the multi skills name for training
            specified_skills=specified_skills,
        )

        super().__init__(
            robot,
            task,
            render=render
        )

