import ray
import wandb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from ray.rllib.utils.filter import MeanStdFilter

from algorithms.worker_ea import EAWorker
from utils_torch.chromosome import VBNChromosome


PLAYER_1_ID = 'robot_0'
PLAYER_2_ID = 'robot_1'


class GAWorker(EAWorker):
    """ Worker class for the Coevolutionary Genetic Algorithm.
    This class handles both the evaluation and mutation of individuals.
    After evaluation, the results are communicated back to the Trainer"""

    def __init__(self, config):
        super().__init__(config)

        self.elite = VBNChromosome(number_actions=self.config['number_actions'])
        self.oponent = VBNChromosome(number_actions=self.config['number_actions'])

        self.filter = MeanStdFilter((14,))

    def evaluate_mutations(self, elite, oponent, record=False, mutate_oponent=True):
        """ Mutate the inputted weights and evaluate its performance against the
        inputted oponent. """
        # recorder = VideoRecorder(self.env, path=self.video_path) if record else None
        if elite != None:
            self.elite.set_weights(elite)
        # else:
        #     self.elite = None
        if oponent != None:
            self.oponent.set_weights(oponent)
        # else:
        #     self.oponent = None

        if mutate_oponent:
            self.oponent.mutate(self.config['mutation_power'])
        elite_reward1, oponent_reward1, ts1 = self.play_game(
            self.elite, self.oponent)
        oponent_reward2, elite_reward2, ts2 = self.play_game(
            self.oponent, self.elite)
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
