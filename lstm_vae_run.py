import copy
import os
import sys

import yaml
import argparse
from pathlib import Path

from sklearn.metrics import mean_squared_error

from models import *
from experiments.lstm_vae_experiment import LSTMVAExperiment
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datasets.time_series import TimeSeriesDataset
from pytorch_lightning.plugins import DDPPlugin

from niapy import Runner
from niapy.problems import Problem
from niapy.algorithms.basic import *
from niapy.algorithms.modified import *


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""
    # https://forums.pytorchlightning.ai/t/how-to-access-the-logged-results-such-as-losses/155

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        each_me = copy.deepcopy(trainer.callback_metrics)
        self.metrics.append(each_me)


parser = argparse.ArgumentParser(description='Generic runner for LSTM VAE models')
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/lstm_vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'], )

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)
metrics = MetricsCallback()
solution = [0.1, 0.220, 0.7, 0.1, 0.1, 0.1, 0.1]  # Symmetrical AE


# solution = [0.1, 0.01, 0.99, 0.1, 0.1, 0.1, 0.1]  # Symmetrical AE

# solution = [0.6, 0.01, 0.7, 0.1, 0.1, 0.1, 0.1]  # Asymmetrical AE
# solution = [0.6, 0.15, 0.37, 0.1, 0.1, 0.1, 0.1]  # Asymmetrical AE


class AutoencoderArchitecture(Problem):

    def __init__(self, dimension, alpha=0.99):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.alpha = alpha
        self.iteration = 0

    def _evaluate(self, solution):
        #solution = [0.15161603,0.50030629,0.51998326,0.48736799,0.71555138,0.97241088,0.99732743]
        #solution = [0.6, 0.15, 0.37, 0.1, 0.1, 0.1, 0.1]
        print("=================================================================================================")
        print(f"ITERATION IS: {self.iteration}")
        print(f"SOLUTION: {solution}")
        self.iteration += 1

        model = vae_models[config['model_params']['name']](solution, **config['model_params'])

        """Punishing bad decisions"""
        if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
            fitness = sys.maxsize
            print(f"Fitness: {fitness}")
            return fitness

        experiment = LSTMVAExperiment(model, config['exp_params'], config['model_params']['n_features'])

        data = TimeSeriesDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

        data.setup()
        runner = Trainer(logger=tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                             monitor="val_loss",
                                             save_last=True),
                             metrics,
                         ],
                         strategy=DDPPlugin(find_unused_parameters=False),

                         **config['trainer_params'])

        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment, datamodule=data)

        # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
        if math.isnan(experiment.val_RMSE.item()):
            fitness = sys.maxsize
            print(f"Fitness: {fitness}")
            return fitness

        else:
            RMSE = experiment.val_RMSE.item()
            complexity = (int(model.epochs) ** 2) + (model.layers * 100) + (model.bottleneck_size * 10)
            fitness = (RMSE * 1000) + (complexity / 100)
            print(f"RMSE: {RMSE}")
            print(f"Complexity: {complexity}")
            print(f"Fitness: {fitness}")
            return fitness





"""
Dimensionality:
y1: topology shape,
y2: number of neurons per layer
y3: number of layers,
y4: activation function
y5: number of epochs,
y6: learning rate
y7: optimizer algorithm.
"""
DIMENSIONALITY = 7


runner = Runner(
    dimension=DIMENSIONALITY,
    max_evals=3,
    runs=1,
    algorithms=[
        ParticleSwarmAlgorithm(),
        DifferentialEvolution(),
        FireflyAlgorithm(),
        SelfAdaptiveDifferentialEvolution(),
        GeneticAlgorithm()
    ],
    problems=[
        AutoencoderArchitecture(DIMENSIONALITY)
    ]
)

print("=================================================================================================")
final_solutions = runner.run(export='json', verbose=True)
best_fitness = sys.maxsize
best_solution = None

for algorithm in final_solutions:
    fitness = final_solutions[algorithm]['AutoencoderArchitecture'][0][1]
    solution = final_solutions[algorithm]['AutoencoderArchitecture'][0][0]
    print(f"{algorithm}'s fitness: {fitness}")
    print(f"{algorithm}'s solution: {solution}")

    if best_fitness > fitness:
        best_fitness = fitness
        best_solution = final_solutions[algorithm]['AutoencoderArchitecture'][0][0]





# for x in range(0, 3):
#     model = vae_models[config['model_params']['name']](solution, **config['model_params'])
#     # config['trainer_params']['max_epochs'] = model.epochs
#     experiment = LSTMVAExperiment(model, config['exp_params'], config['model_params']['n_features'])
#
#     data = TimeSeriesDataset(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)
#
#     data.setup()
#     runner = Trainer(logger=tb_logger,
#                      callbacks=[
#                          LearningRateMonitor(),
#                          ModelCheckpoint(save_top_k=2,
#                                          dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
#                                          monitor="val_loss",
#                                          save_last=True),
#                      ],
#                      strategy=DDPPlugin(find_unused_parameters=False),
#
#                      **config['trainer_params'])
#
#     Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
#     Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)
#
#     print(f"======= Training {config['model_params']['name']} =======")
#     runner.fit(experiment, datamodule=data)
#
#     torch.save(model, f"LSTMVAE_model_{config['trainer_params']['max_epochs']}_{x}_epochs.pt")
