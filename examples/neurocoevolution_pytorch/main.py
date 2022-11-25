from pathlib import Path

import yaml
from ray import tune
from ray.tune.integration.wandb import WandbLogger
from algorithms.trainer_ga import GATrainer
config = {}
with open('configs/config_ga_test.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

trainer = GATrainer(config)

while trainer.timesteps() < config['stop_criteria']['timesteps_total']:
    trainer.step()
