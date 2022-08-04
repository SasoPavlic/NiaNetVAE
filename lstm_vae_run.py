import math
import os
import sys
import yaml
import argparse

from models import *
from experiments.lstm_vae_experiment import LSTMVAExperiment
from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from datasets.time_series import TimeSeriesDataset
from pytorch_lightning.plugins import DDPPlugin

from niapy import Runner
from niapy.problems import Problem
from niapy.algorithms.basic import *
from niapy.algorithms.modified import *

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

early_stop_callback = EarlyStopping(monitor=config['early_stop']['monitor'],
                                    min_delta=config['early_stop']['min_delta'],
                                    patience=config['early_stop']['patience'],
                                    verbose=False,
                                    check_finite=True,
                                    mode="max")
# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)


class VariationalAutoencoderArchitecture(Problem):

    def __init__(self, dimension):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.iteration = 0

    def _evaluate(self, solution):
        # solution = [0.18068983, 0.05792889, 0.55358249, 0.3777263, 0.57080761, 0.67469747, 0.49576287]

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
        config['trainer_params']['max_epochs'] = model.epochs
        data.setup()

        runner = Trainer(logger=tb_logger,
                         callbacks=[
                             LearningRateMonitor(),
                             ModelCheckpoint(save_top_k=2,
                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                             monitor="val_loss",
                                             save_last=True),
                             early_stop_callback,
                         ],
                         strategy=DDPPlugin(find_unused_parameters=False),

                         **config['trainer_params'])

        print(f"======= Training {config['model_params']['name']} =======")

        print(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
        runner.fit(experiment, datamodule=data)
        print(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')

        # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
        if math.isnan(experiment.val_RMSE.item()):
            fitness = sys.maxsize
            print(f"Fitness: {fitness}")
            return fitness

        else:
            RMSE = experiment.val_RMSE.item()
            complexity = (model.epochs ** 2) + (model.layers * 100) + (model.bottleneck_size * 10)
            fitness = (RMSE * 1000) + (complexity / 100)
            print(f"RMSE: {RMSE}")
            print(f"Complexity: {complexity}")
            print(f"Fitness: {fitness}")
            return fitness


if __name__ == '__main__':
    print(f'Program start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
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
        max_evals=5,
        runs=1,
        algorithms=[
            ParticleSwarmAlgorithm(),
            DifferentialEvolution(),
            FireflyAlgorithm(),
            SelfAdaptiveDifferentialEvolution(),
            GeneticAlgorithm()
        ],
        problems=[
            VariationalAutoencoderArchitecture(DIMENSIONALITY)
        ]
    )

    print("=================================================================================================")
    final_solutions = runner.run(export='json', verbose=True)
    best_fitness = sys.maxsize
    best_solution = None

    for algorithm in final_solutions:
        fitness = final_solutions[algorithm]['VariationalAutoencoderArchitecture'][0][1]
        solution = final_solutions[algorithm]['VariationalAutoencoderArchitecture'][0][0]
        print(f"{algorithm}'s fitness: {fitness}")
        print(f"{algorithm}'s solution: {solution}")

        if best_fitness > fitness:
            best_fitness = fitness
            best_solution = final_solutions[algorithm]['VariationalAutoencoderArchitecture'][0][0]

    best_model = vae_models[config['model_params']['name']](best_solution, **config['model_params'])
    torch.save(best_model, f"LSTMVAE_model_{config['trainer_params']['max_epochs']}_epochs.pt")

    end = datetime.now().strftime("%H:%M:%S-%d/%m/%Y")
    print(f"\n Program end: {end}")
