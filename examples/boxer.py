import gym
import urdfenvs.boxer_robot
import numpy as np


def run_boxer(n_steps=1000, render=False, goal=True, obstacles=True):
    env = gym.make("boxer-robot-vel-v0", dt=0.01, render=render)
    action = np.array([0.6, 0.8])
    cumReward = 0.0
    pos0 = np.array([1.0, 0.2, -1.0]) * 0.0
    ob = env.reset(pos=pos0)
    print(f"Initial observation : {ob}")
    env.add_walls()
    print("Starting episode")
    history = []
    for _ in range(n_steps):
        ob, _, _, _ = env.step(action)
        history.append(ob)
    return history


if __name__ == "__main__":
    run_boxer(render=True)
