import gymnasium as gym
from gymnasium import space
import numpy as np

class SwingUpInvPenWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, 
              seed: int | None = None,
              options: dict = None):
        obs, info = self.env.reset()