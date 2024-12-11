from enum import Enum
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from gymnasium_envs.envs.core import RobotTaskEnv
from gymnasium_envs.envs.robots.UR5e import dualUR5e, singleUR5e, ReachsingleUR5e
from gymnasium_envs.envs.tasks.kitchen import KitchenMultiTask
from gymnasium_envs.MJFunc import MJFunc


class ReachEnv_v0(RobotTaskEnv):
    def __init__(self,
                 render: bool = True,
                 xml_path: str = '',
                 xml_file_name: str = '',
                 basic_skills: list = [],
                 specified_skills: list = [],
                 kitchen_tasks_name: list = [],
                 kitchen_tasks_chain: dict = {},
                 normalization_range: list = [-1, 1],
                 ):

        sim = MJFunc(
            render=render,
            xml_path=xml_path,
            xml_file_name=xml_file_name,
        )

        robot = ReachsingleUR5e(
            sim=sim,
            normalization_range=normalization_range,
        )

        task = KitchenMultiTask(
            sim=sim,
            basic_skills=basic_skills,  # the multi skills name for training
            specified_skills=specified_skills,
            kitchen_tasks_name=kitchen_tasks_name,
            kitchen_tasks_chain=kitchen_tasks_chain,
            normalization_range=normalization_range,
        )

        super().__init__(
            robot,
            task,
            render=render,
            normalization_range=normalization_range,
        )

