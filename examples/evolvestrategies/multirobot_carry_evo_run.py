import tqdm as tqdm
from torch.multiprocessing import Pool
from torch.optim import Adam

from evostrat import compute_centered_ranks, NormalPopulation
from MRCarry import MRCarry
import wandb
import pickle
import torch
from typing import Dict
import gym
from torch import nn
import torch as t
from evostrat import Individual
import numpy as np

from urdfenvs.robots.generic_urdf import GenericUrdfReacher


def flatten_observation(observation_dictonary: dict) -> np.ndarray:
    observation_list = []
    for val in observation_dictonary.values():
        if isinstance(val, np.ndarray):
            observation_list += val.tolist()
        elif isinstance(val, dict):
            observation_list += flatten_observation(val).tolist()
    observation_array = np.array(observation_list)
    return observation_array


def make_env(render=False):
    robots = [
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
        GenericUrdfReacher(urdf="/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/loadPointRobot.urdf", mode="vel"),
    ]
    env = gym.make("urdf-env-v1", dt=0.01, robots=robots, render=render)
    # Choosing arbitrary actions
    base_pos = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    env.reset(base_pos=base_pos)
    env.add_stuff()
    ob = env.get_observation()
    return env, ob

def kill_env(env):
    env.close()
    del env


state_params = pickle.load(open('/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/evolvestrategies/best_model.p', 'rb'))


mrc1 = MRCarry()
mrc1.net.load_state_dict(state_params)
import numpy as np
import pybullet as p

env, obs = make_env(render=True)
p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "/home/josyula/Programs/MAS_Project/gym_envs_urdf/examples/evolvestrategies/evo_passage1.mp4")
obs = flatten_observation(obs)
done = False
r_tot = 0
alpha = 0.5
action_prev = np.zeros((6,))
rf = 0
while not done:
    action = mrc1.net(torch.FloatTensor(obs))
    actions = np.clip(np.hstack((action.detach().numpy().reshape(2, 2), np.zeros((2, 1)))).ravel(), -0.5, 0.5)
    action_new = alpha * actions + (1 - alpha) * action_prev
    action_prev = action_new
    obs, rew, done, info = env.step(action_new)
    r_tot += (rew[0] + rew[1]) * 0.5 + rf * 10
    obs = flatten_observation(obs)
    if info['goal_reached']:
        print("Goal Reached")
        break

p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
kill_env(env)
