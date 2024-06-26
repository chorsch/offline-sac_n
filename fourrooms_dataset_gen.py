import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from os import sys
sys.path.append('/Users/caroline/Desktop/projects/repos/')
from four_room.env import FourRoomsEnv
from four_room.wrappers import gym_wrapper
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)


num_steps = 1000

def wrap_env(
    env: gym.Env,
    state_mean: Union[np.ndarray, float] = 0.0,
    state_std: Union[np.ndarray, float] = 1.0,
    reward_scale: float = 1.0,
) -> gym.Env:
    def normalize_state(state):
        return (state - state_mean) / state_std

    def scale_reward(reward):
        return reward_scale * reward

    env = gym.wrappers.TransformObservation(env, normalize_state)
    if reward_scale != 1.0:
        env = gym.wrappers.TransformReward(env, scale_reward)
    return env


def get_random_dataset():
    env = wrap_env(gym_wrapper(gym.make(('MiniGrid-FourRooms-v1'))))
    observation, info = env.reset()

    dataset = {'observations':[], 'next_observations':[], 'actions':[], 'rewards':[],
                'terminals':[], 'timeouts':[], 'infos':[]}
    for i in range(num_steps):
        action = env.action_space.sample()
        last_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

        dataset['observations'].append(np.array(last_observation).flatten())
        dataset['next_observations'].append(np.array(observation).flatten())
        dataset['actions'].append(np.array([action]))
        dataset['rewards'].append(reward)
        dataset['terminals'].append(terminated)
        dataset['timeouts'].append(truncated)
        dataset['infos'].append(info)

    for key in dataset:
        dataset[key] = np.array(dataset[key])
    return dataset
