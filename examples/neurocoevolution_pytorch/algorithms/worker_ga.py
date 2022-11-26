import ray
import wandb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.utils.filter import MeanStdFilter

from algorithms.worker_ea import EAWorker
from utils_torch.chromosome import VBNChromosome
import numpy as np


PLAYER_1_ID = 'robot_0'
PLAYER_2_ID = 'robot_1'

class GAWorker(EAWorker):
    """ Worker class for the Coevolutionary Genetic Algorithm.
    This class handles both the evaluation and mutation of individuals.
    After evaluation, the results are communicated back to the Trainer"""

    def __init__(self, config):
        super().__init__(config)

        self.player1 = VBNChromosome(number_actions=self.config['number_actions'])
        self.player2 = VBNChromosome(number_actions=self.config['number_actions'])

        self.filter = MeanStdFilter((14,2))
        self.wall_pass = 0
        self.goal_reach = 0
    #mutate individual get weights
    def mutate_individual(self, individual):
        """ Mutate the inputted weights and evaluate its performance against the
        inputted oponent. """
        mutation_power = self.config['mutation_power']
        weights = individual
        for key in weights.keys():
            noise = np.random.normal(loc=0.0, scale=mutation_power, size=weights[key].shape)
            weights[key] = weights[key] + noise
        return weights

    def evaluate_team_fitness(self, player1, player2):
        self.player1.set_weights(player1)
        self.player2.set_weights(player2)
        reward1 =0
        reward2 =0
        for i in range(3):
            reward_1, reward_2, ts, wall_pass, goal_reach = self.play_game(self.player1, self.player2)
            self.wall_pass += wall_pass
            self.goal_reach += goal_reach
            reward1 += reward_1
            reward2 += reward_2
        return (reward1)/3.

    def evaluate_mutations(self, elite, oponent):
        """ Mutate the inputted weights and evaluate its performance against the
        inputted oponent. """
        # recorder = VideoRecorder(self.env, path=self.video_path) if record else None
        self.elite.set_weights(elite)
        self.oponent.set_weights(oponent)

        elite_reward1, oponent_reward1, ts1, wall_pass, goal_reach = self.play_game(self.elite, self.oponent)
        oponent_reward2, elite_reward2, ts2 = self.play_game(self.oponent, self.elite)
        total_elite = elite_reward1 + elite_reward2
        total_oponent = oponent_reward1 + oponent_reward2
        # if record:
        #     recorder.close()
        #
        return {
            'oponent_weights': self.oponent.get_weights() if self.oponent else None,
            'score_vs_elite': total_oponent,
            'timesteps_total': ts1 + ts2,
            'video': None
        }
