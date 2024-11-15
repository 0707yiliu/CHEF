import numpy as np
import math
from gymnasium_envs.envs.core import MJRobot

class dualUR5e(MJRobot):
    def __init__(self,
                 sim) -> None:
        self.sensor_num = 4
        super().__init__(
            sim,
        )

