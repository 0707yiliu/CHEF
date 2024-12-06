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
                 init_qpos: np.ndarray,
                 joint_list: list,
                 actuator_list: list,
                 sensor_list: list,
                 ) -> None:
        self.sim = sim

        self.init_qpos = init_qpos
        self.actuator_list = actuator_list
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
    def reset(self, env_index, task_result):
        """Reset the robot and return the observation."""

    @abstractmethod
    def compute_reward(self):
        """Compute the reward wit DMPs as the demonstration traj in robot part"""

    def setup(self) -> None:
        """Called after robot loading."""
        pass

class Task(ABC):
    def __init__(self,
                 sim) -> None:
        self.sim = sim
        self.goal = None

    @abstractmethod
    def reset(self, env_index):
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
    def compute_reward(self) -> Union[np.ndarray, float]:
        """Compute reward associated to the achieved and the desired goal."""

    def get_desired_goal(self):
        """return the current desired goal"""
        if self.goal is None:
            raise RuntimeError(("No goal yet, call reset() to select one task for getting goal"))
        else:
            return self.goal.copy()

class RobotTaskEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self,
                 robot: MJRobot,
                 task: Task,
                 render: bool,
                 normalization_range: list = [0, 1],
                 ) -> None:
        assert robot.sim == task.sim, "The robot and the task must belong to the same simulation."
        self.robot = robot
        self.task = task
        self.render = render
        self.action_space = self.robot.action_space
        obs = self._get_obs()
        _common_obs_shape = obs['common_observation'].shape[0]
        _current_state_shape = obs['current_state'].shape[0]
        _desired_state_shape = obs['desired_state'].shape[0]
        norm_max = normalization_range[1]
        norm_min = normalization_range[0]
        self.observation_space = spaces.Dict(
            {
                "common_observation": spaces.Box(norm_min, norm_max, shape=(_common_obs_shape,), dtype=np.float32),
                "current_state": spaces.Box(norm_min, norm_max, shape=(_current_state_shape,), dtype=np.float32),
                "desired_state": spaces.Box(norm_min, norm_max, shape=(_desired_state_shape,), dtype=np.float32),
            }
        )

    def reset(self, seed: Optional[int] = None, options={}):
        super().reset(seed=seed)
        _reset_goal = True
        # reset the env first, because the robot with different skill need to be reset to different state
        # the mujoco reset would be used in robot.reset()
        reset_skill_num = np.random.randint(0, 3)
        reset_skill_num = 0  # for debug the reset of each environment
        while _reset_goal is True:
            task_result = self.task.reset(reset_skill_num)
            _reset_goal = self.robot.reset(reset_skill_num, task_result)
        return self._get_obs()

    def _get_obs(self):
        robot_obs = self.robot.get_obs()
        task_obs = self.task.get_obs()
        observation = np.concatenate([robot_obs, task_obs])
        current_state = self.task.get_achieved_goal()
        desired_state = self.task.get_desired_goal()
        return {
            'common_observation': observation,
            'current_state': current_state,
            'desired_state': desired_state,
        }
        # return {
        #     'common_observation': np.array([-1, -1]),
        #     'current_state': np.array([-1, -1]),
        #     'desired_state': np.array([-1, -1]),
        # } # !for test

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array([1,2,3]) - np.array([1,3,3]), ord=1
            )
        }

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.robot.set_action(action)
        obs = self._get_obs()
        terminated = False
        reward = self.robot.compute_reward()
        reward = 1 if terminated else 0
        info = self._get_info()
        return obs, reward, terminated, False, info


