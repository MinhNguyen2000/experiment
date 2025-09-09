import gymnasium as gym
import numpy as np

class SwingUpInvPenWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        core = self.env.unwrapped
        self.x_threshold = 2
        self.observation_space

    def reset(self, **kwargs):
        
        # --- reset the environment and unwrap the TimeLimit wrapper
        obs, info = self.env.reset(**kwargs)

        core = self.env.unwrapped

        # # --- set pendulum down initial position with some randomness
        # state = np.array([0.0, np.pi, 0.0, 0.0], dtype=np.float32)

        # state += core.np_random.normal(0, 0.1, size=4)

        # core.state = tuple(state)
        qpos = np.array([0.0, np.pi]) + core.np_random.normal(0, 0.1, size=2)
        qvel = np.array([0.0, 0.0]) + core.np_random.normal(0, 0.1, size=2)
        core.set_state(qpos,qvel)

        obs = np.array(core._get_obs(), dtype = np.float32)

        return obs, info
    
    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(action)
        
        core = self.env.unwrapped

        # --- unpack state
        x, theta, x_dot, theta_dot = core._get_obs()
        print()

        # --- termination when cartpos out of bound
        term = bool(abs(x) > self.x_threshold)

        # reward_theta is 1 when theta is 0 or 2pi, 0 if between 90 and 270:
        reward_theta = max(0, np.cos(theta))

        # reward_x is 0 when cart is at the edge of the screen, 1 when it's in the center:
        reward_x = np.cos((x / self.x_threshold) * (np.pi / 2.0))

        # reward between [0, 1]:
        rew = reward_theta * reward_x

        return obs, rew, term, trunc, info