from abc import abstractmethod, ABCMeta
import wandb
from customenv import *
import torch
PLAYER_1_ID = "robot_0"
PLAYER_2_ID = "robot_1"




class EAWorker:
    """ Class that includes some functionality that is used by both the
    Evolution Strategies and Genetic Algorithm Workers. """

    __metaclass__ = ABCMeta

    def __init__(self,
                 config):

        self.config = config
        print(f"Hello world from worker")

    def evaluate(self, weights):
        """ Evlauate weights by playing against a random policy. """
        # recorder = VideoRecorder(self.env, path=self.video_path_eval)
        self.elite.set_weights(weights)
        reward, _, ts = self.play_game(self.elite,
                                       None,
                                       recorder=None,
                                       eval=True)
        return {
            'total_reward': reward,
            'timesteps_total': ts,
            'video': wandb.Video(self.video_path_eval),
        }

    def play_game(self, player1, player2, recorder=None, eval=False):
        """ Play a game using the weights of two players. """
        env, obs = make_env(render=False)
        reward1 = 0
        reward2 = 0
        limit = self.config['max_evaluation_steps'] if eval else self.config[
            'max_timesteps_per_episode']
        for ts in range(limit):
            filtered_obs1 = flatten_observation(obs[PLAYER_1_ID])
            filtered_obs2 = flatten_observation(obs[PLAYER_2_ID])
            filtered_obs1 = torch.FloatTensor(filtered_obs1)  #.to(self.device)
            filtered_obs2 = torch.FloatTensor(filtered_obs2)
            action1 = torch.FloatTensor(np.random.randn(2,)) if player1==None else player1.determine_actions(filtered_obs1)
            action2 = torch.FloatTensor(np.random.randn(2,)) if player2==None else player2.determine_actions(filtered_obs2)
            actions = np.stack([action1.detach().cpu().numpy(), action2.detach().cpu().numpy()])
            actions = np.hstack((actions.reshape(2, 2), np.zeros((2, 1)))).ravel()
            obs, reward, done, info = env.step(actions)
            reward1 += reward[0]
            reward2 += reward[1]
            if done:
                break
        kill_env(env)
        return reward1, reward2, ts

