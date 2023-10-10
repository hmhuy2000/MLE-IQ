import gym
from stable_baselines3.common.monitor import Monitor

import gym
from gym import spaces
import numpy as np

def check_and_normalize_box_actions(env):
    if isinstance(env.action_space, spaces.Box):
        low, high = env.action_space.low, env.action_space.high
        if (np.abs(low + np.ones_like(low)).max() > 1e-6 or
                np.abs(high - np.ones_like(high)).max() > 1e-6):
            print('--> Normalizing environment actions.')
            raise
    return env

def make_env(args, monitor=True):
    env = gym.make(args.env.name)
    if monitor:
        env = Monitor(env, "gym")

    env = check_and_normalize_box_actions(env)
    return env