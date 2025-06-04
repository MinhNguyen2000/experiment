''' This file is used for customizing the frozen lake environment in the gymnasium package

This is done using wrappers to safely modify the environment without altering the underlying code

TO-DO:
1. Modify the reward structure to have penalties at the holes
2. Incorporate exploring start, such that env.reset() put the agent in a random non-terminal state'''

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.registration import register


class CustomFrozenLakeEnv(FrozenLakeEnv):

    def step(self, action):
        obs, _, terminated, truncated, info = super().step(action)
        
        # Get the current state's tile type
        current_tile = self.desc.flatten()[obs]
        
        # Custom reward structure
        if current_tile == b'G':
            reward = 10
        elif current_tile == b'H':
            reward = -2
        else:
            reward = -0.1  # Small penalty for each step
        
        return obs, reward, terminated, truncated, info

register(
    id="CustomFrozenLake-v1",
    entry_point="custom_frozenlake_2:CustomFrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)

