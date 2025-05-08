''' This file is used for customizing the frozen lake environment in the gymnasium package

This is done using wrappers to safely modify the environment without altering the underlying code

TO-DO:
1. Modify the reward structure to have penalties at the holes
2. Incorporate exploring start, such that env.reset() put the agent in a random non-terminal state'''

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
from gymnasium.envs.registration import register


class CustomFrozenLakeEnv(FrozenLakeEnv):
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.nS = self.nrow * self.ncol
        

        # Build once outside the loop
        terminal_states = set(
            s for s in range(self.nS)
            if self.desc.flat[s] in (b'H', b'G')
        )

        # Sample non-terminal states
        while True:
            # self.s = categorical_sample(self.initial_state_distrib, self.np_random)
            self.s = self.np_random.choice(self.nS)
            if self.s not in terminal_states: 
                break
        
        self.lastaction = None

        if self.render_mode == "human":
            self.render()

        return int(self.s), {"prob": 1}

register(
    id="CustomFrozenLake-v0",
    entry_point="custom_frozenlake:CustomFrozenLakeEnv",
    kwargs={"map_name": "4x4", "is_slippery": True},
)

