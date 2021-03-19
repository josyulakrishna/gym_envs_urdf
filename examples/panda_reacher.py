import gym
import pandaReacher
import time
import numpy as np


def main():
    env = gym.make('panda-reacher-tor-v0', dt=0.01, render=True)
    n_episodes = 1
    n_steps = 1000
    cumReward = 0.0
    for e in range(n_episodes):
        ob = env.reset()
        print("Starting episode")
        for i in range(n_steps):
            time.sleep(env._dt)
            action = env.action_space.sample()
            ob, reward, done, info = env.step(action)
            cumReward += reward


if __name__ == '__main__':
    main()