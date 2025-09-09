import gymnasium as gym
import numpy as np

class SwingUpInvPenWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        core = self.env.unwrapped
        self.x_threshold = 2

    def reset(self, **kwargs):
        
        # --- reset the environment and unwrap the TimeLimit wrapper
        obs, info = self.env.reset(**kwargs)

        core = self.env.unwrapped

        # --- set pendulum down initial position with some randomness
        state = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)

        state += core.np_random.normal(0, 0.1, size=4)

        core.state = tuple(state)

        obs = np.array(core.state, dtype = np.float32)

        return obs, info
