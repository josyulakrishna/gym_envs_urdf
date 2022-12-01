from pathlib import Path

import yaml
# from algorithms.trainer_ga import GATrainer
from algorithms.trainer_es import ESTrainer
config = {}
import torch.multiprocessing as mp

mp.set_start_method('spawn', force=True)

if __name__ == '__main__':
    with open('configs/config_ga_test.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # trainer = GATrainer(config)
    trainer = ESTrainer(config)
    while trainer.generation < config['stop_criteria']['generations']:
        # trainer.step_hof()
        trainer.step()