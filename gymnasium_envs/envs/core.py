from typing import Any, SupportsFloat, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, ActType

from abc import ABC, abstractmethod

class MJRobot(ABC):
    def __init__(self,
                 sim,
                 action_space: spaces.Space,
                 joint_index: np.ndarray,
                 joint_force: np.ndarray,
                 joint_list: list,
                 sensor_list: list,
                 ) -> None:
        self.sim = sim
        self.action_space = action_space
        self.joint_index = joint_index
        self.joint_force = joint_force
        self.joint_list = joint_list
        self.sensor_list = sensor_list

    @abstractmethod
    def set_action(self, action: np.ndarray) -> None:
        """Set the action. Must be called just before sim.step().
        Args:
            action (np.ndarray): The action.
        """
    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the robot.

        Returns:
            np.ndarray: The observation.
        """
    @abstractmethod
    def reset(self) -> None:
        """Reset the robot and return the observation."""
    def setup(self) -> None:
        """Called after robot loading."""
        pass

    def get_body_position(self, body: str) -> np.ndarray:
        return self.sim.get_body_position(body=body)

class Task(ABC):
    def __init__(self,
                 sim) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self) -> None:
        """Reset the task: sample a new goal."""

    @abstractmethod
    def get_obs(self) -> np.ndarray:
        """Return the observation associated to the task."""

    @abstractmethod
    def get_achieved_goal(self) -> np.ndarray:
        """Return the achieved goal."""

    @abstractmethod
    def is_success(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        """Returns whether the achieved goal match the desired goal."""

    @abstractmethod
    def compute_reward(
            self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info: Dict[str, Any] = {}
    ) -> Union[np.ndarray, float]:
        """Compute reward associated to the achieved and the desired goal."""

    @abstractmethod
    def get_desired_goal(self):
        """return the current desired goal"""
        if self.goal is None:
            raise RuntimeError(("No goal yet, call reset() to select one task for getting goal"))
        else:
            return self.goal.cope()

    def get_site_position(self, body: str) -> np.ndarray:
        return self.sim.get_site_position(body=body)

class RobotTaskEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self,
                 robot: MJRobot,
                 task: Task,
                 render: bool) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.sim = robot.sim
        self.robot = robot
        self.task = task
        self.render = render
        self.action_space = self.robot.action_space
        self.observation_space = spaces.Box(-1, 1, shape=(2,), dtype=np.float32)
        self._get_obs()

    def reset(self):
        pass

    def _get_obs(self):
        # robot_obs = self.robot.get_obs()
        # task_obs = self.task.get_obs()
        # observation = np.concatenate([robot_obs, task_obs])
        # current_state = self.task.get_achieved_goal()
        # desired_state = self.task.get_desired_goal()
        # self.observation_space = spaces.Dict(
        #     {
        #         "common_obs": spaces.Box(-1, 1, shape=observation.shape, dtype=np.float32),
        #         'current_state': spaces.Box(-1, 1, shape=current_state.shape, dtype=np.float32),
        #         'desired_state': spaces.Box(-1, 1, shape=desired_state.shape, dtype=np.float32),
        #     }
        # )
        pass

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass

