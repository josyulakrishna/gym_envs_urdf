# from pathlib import Path

import numpy as np
# import ray
import yaml
# from ray.rllib.agents import with_common_config
# from torch.multiprocessing.pool import Pool
from algorithms.trainer_ea import EATrainer
from utils_torch.chromosome import VBNChromosome

DEFAULT_CONFIG = {}
import wandb
wandb.init(project="neurocoevolution_mr_carry", entity="josyula", sync_tensorboard=True, config=DEFAULT_CONFIG)
with open('configs/config_ga_test.yaml') as f:
    DEFAULT_CONFIG = yaml.load(f, Loader=yaml.FullLoader)



class GATrainer(EATrainer):
    _name = "GA"
    _default_config = DEFAULT_CONFIG

    def __init__(self, config):
        """ Trainer class for the Coevolutionary Genetic Algorithm.
        This class distributes the mutation and evaluation workload over a number
        of workers and updates and maintains the population."""

        super(GATrainer, self).__init__(config)

        self.elites = [VBNChromosome(number_actions=self.config['number_actions'])
                       for _ in range(config['number_elites'])]
        # samples = self.collect_samples()
        # for chrom in self.elites:
        #     chrom.virtual_batch_norm(samples)

        self.hof = [self.elites[i].get_weights() for i in
                    range(self.config['number_elites'])]
        self.winner = None


    def step(self):
        """ Evolve the next generation using the Genetic Algorithm. This process
        consists of three steps:
        1. Communicate the elites of the previous generation
        to the workers and let them mutate and evaluate them against individuals from
        the Hall of Fame. To include a form of Elitism, not all elites are mutated.
        2. Communicate the mutated weights and fitnesses back to the trainer and
        determine which of the individuals are the fittest. The fittest individuals
        will form the elites of the next population.
        3. Evaluate the fittest
        individual against a random policy and log the results. """

        # Evaluate mutations vs first hof
        # worker_jobs = []
        player1 = []
        player2 = []
        should_record_ =[]
        should_mutate_ = []
        for i in range(self.config['population_size']):
            # worker_id = i % self.config['num_workers']
            elite_id = i % self.config['number_elites']
            should_mutate = (i > self.config['number_elites'])
            should_record = False
            player1.append(self.hof[-1])
            player2.append(self.elites[elite_id].get_weights())
            should_record_.append(should_record)
            should_mutate_.append(should_mutate)

        args = zip(player1, player2, should_record_, should_mutate_)
        results = self._workers.starmap(self.worker_class.evaluate_mutations, args)
        # results = self.worker_class.evaluate_mutations( player1[-1], player2[-1], should_record_[-1], should_mutate_[-1])
        # Evaluate vs other hof
        player1 = []
        player2 = []
        should_record_ =[]
        should_mutate_ = []

        for j in range(len(self.hof) - 1):
            for i in range(self.config['population_size']):
                # worker_id = len(worker_jobs) % self.config['num_workers']
                player1.append(self.hof[-2 - j])
                player2.append(results[i]['oponent_weights'])
                should_record_.append(False)
                should_mutate_.append(False)

        args = zip(player1, player2, should_record_, should_mutate_)

        results += self._workers.starmap(self.worker_class.evaluate_mutations, args)

        rewards = []
        print(len(results))
        for i in range(self.config['population_size']):
            total_reward = 0
            for j in range(self.config['number_elites']):
                reward_index = self.config['population_size'] * j + i
                total_reward += results[reward_index]['score_vs_elite']
            rewards.append(total_reward)

        best_mutation_id = np.argmax(rewards)
        best_mutation_weights = results[best_mutation_id]['oponent_weights']
        print(f"Best mutation: {best_mutation_id} with reward {np.max(rewards)}")
        if np.max(rewards) > 100:
            print("Winner found!")
            self.try_save_winner(best_mutation_weights)
        self.hof.append(best_mutation_weights)

        new_elite_ids = np.argsort(rewards)[-self.config['number_elites']:]
        print(f"TOP mutations: {new_elite_ids}")
        for i, elite in enumerate(self.elites):
            mutation_id = new_elite_ids[i]
            elite.set_weights(results[mutation_id]['oponent_weights'])

        # Evaluate best mutation vs random agent

        evaluate_results = self.evaluate_current_weights(best_mutation_weights)
        evaluate_rewards = [result['score_vs_elite'] for result in evaluate_results]

        train_rewards = [result['score_vs_elite'] for result in results]
        # evaluate_videos = [result['video'] for result in evaluate_results]

        self.increment_metrics(results)

        summary = {
            "timesteps_total":self.timesteps_total,
            "episodes_total":self.episodes_total,
            "train_reward_min": np.min(train_rewards),
            "train_reward_mean": np.mean(train_rewards),
            "train_reward_med": np.median(train_rewards),
            "train_reward_max": np.max(train_rewards),
            "train_top_5_reward_avg": np.mean(np.sort(train_rewards)[-5:]),
            "evaluate_reward_min": np.min(evaluate_rewards),
            "evaluate_reward_mean": np.mean(evaluate_rewards),
            "evaluate_reward_med": np.median(evaluate_rewards),
            "evaluate_reward_max": np.max(evaluate_rewards),
            "avg_timesteps_train": np.mean(
                [result['timesteps_total'] for result in results]),
            "avg_timesteps_evaluate": np.mean(
                [evaluate_results['timesteps_total']]),
            "eval_max_video": 0,
            "eval_min_video": 0,
            "total_timesteps" : self.timesteps_total,
        }
        wandb.log(summary)

        # self.add_videos_to_summary(results, summary)
        return summary

    def timesteps(self):
        return self.timesteps_total