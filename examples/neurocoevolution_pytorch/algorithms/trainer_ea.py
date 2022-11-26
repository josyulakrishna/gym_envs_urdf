import os
from abc import ABCMeta, abstractmethod

import numpy as np
import ray
# from ray.rllib.agents import Trainer, with_common_config
# from ray.rllib.utils.annotations import override
from ray.rllib.utils.filter import MeanStdFilter
from torch.multiprocessing import Pool, Process, set_start_method

from algorithms.worker_ga import GAWorker
from utils_torch.customenv import *
import pickle

PLAYER_1_ID = 'robot_0'
PLAYER_2_ID = 'robot_1'

class EATrainer():
    """ Class that includes some functionality that is used by both the
    Evolution Strategies and Genetic Algorithm Trainers. """
    __metaclass__ = ABCMeta

    def __init__(self, config):
        self.config = config
        self.worker_class = GAWorker(config)
        self._workers = Pool(config["num_workers"])

        self.episodes_total = 0
        self.timesteps_total = 0
        self.generation = 0

    def collect_samples(self):
        """ Sample game frames from the environment by letting two random policies
        play against eachother. """
        env,obs = make_env()
        obs_filter = MeanStdFilter(flatten_observation(obs[PLAYER_1_ID]).shape)
        policy = np.clip(np.random.randn(2 * 2), -0.5, 0.5)
        policy = np.hstack((policy.reshape(2, 2), np.zeros((2, 1)))).ravel()
        samples = []
        for _ in range(500):
            obs, _, done, _ = env.step(policy)
            samples += [obs_filter(flatten_observation(obs[PLAYER_1_ID])), obs_filter(flatten_observation(obs[PLAYER_2_ID]))]
            if done:
                kill_env(env)
                env, _ = make_env()
        return samples

    @abstractmethod
    def step(self):
        """ Should be overwritten by child class. """
        pass

    def try_save_winner(self, winner_weights):
        """ Save the best weights to a file. """
        if not os.path.exists('results'):
            os.mkdir('results')
        filename = f'results/winner_weights_generation_{self.generation}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(winner_weights, f)
        # filename = f'/tmp/winner_weights_generation_{self.generation}.npy.mp4'
        # with open(filename, 'wb+') as file:
        #     np.save(file, winner_weights)
        return filename

    def add_videos_to_summary(self, results, summary):
        """ Add videos to the summary dictionary s.t. they can be logged to the wandb
        framework. """
        for i, result in enumerate(results):
            video = result['video']
            if video:
                summary[f'train_video_{i}'] = results[i]['video']

    def evaluate_current_weights(self, best_mutation_weights):
        """ Send the weights to a number of workers and ask them to evaluate the weights. """
        evaluate_results = [self.worker_class.evaluate_mutations(best_mutation_weights, None, mutate_oponent=False) for i in range(20)]
        return evaluate_results

    def increment_metrics(self):
        """ Increment the total timesteps, episodes and generations. """
        # self.timesteps_total += sum([result['timesteps_total'] for result in results])
        # self.episodes_total += len(results)
        self.generation += 1
