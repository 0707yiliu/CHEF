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
                 sensor_list: list) -> None:
        self.sim = sim

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
        print('here')

    def reset(self):
        pass

    def _get_obs(self):
        pass

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        pass


