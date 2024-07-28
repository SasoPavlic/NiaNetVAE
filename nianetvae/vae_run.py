import argparse
import math
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from lightning.pytorch import Trainer
# from lightning.pytorch.plugins import DDPPlugin
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution, FireflyAlgorithm, GeneticAlgorithm
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from niapy.task import OptimizationType
from tabulate import tabulate

from log import Log
from nianetvae.dataloaders.ecg_dataloader import ECG5000DataLoader
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment, FineTuneLearningRateFinder
from nianetvae.models.rnn_vae import RNNVAE
from nianetvae.niapy_extension import *
from nianetvae.niapy_extension.wrapper import ExtendedProblem, ExtendedRunner
from nianetvae.storage.database import SQLiteConnector

RUN_UUID = None
config = None
conn = None
datamodule = None


def calculate_fitness(model, experiment):
    if experiment.metrics.are_metrics_complete():

        error_x = experiment.metrics.MSE + experiment.metrics.RMSE + experiment.metrics.MAE + experiment.metrics.ABS_REL + experiment.metrics.LOG10
        error_y = experiment.metrics.DELTA1 + experiment.metrics.DELTA2 + experiment.metrics.DELTA3

        C_LAYERS = 10000
        C_BOTTLENECK = 1000

        max_layers, min_layers = config['data_params']['horizontal_dim'], 0
        max_bottleneck, min_bottleneck = config['data_params']['horizontal_dim'], 0

        normalized_num_layers = experiment.metrics.normalize(len(model.encoding_layers), min_layers, max_layers)
        normalized_bottleneck = experiment.metrics.normalize(model.bottleneck_size, min_bottleneck, max_bottleneck)

        complexity = (normalized_num_layers * C_LAYERS) + (normalized_bottleneck * C_BOTTLENECK)
        error = error_x - error_y

        fitness = error + complexity
        return fitness, error, complexity
    else:
        Log.error("Some metric values are still None.")
        return int(9e10), int(9e10), int(9e10)


def upload_save_model(alg_name, iteration, solution, error, model, experiment, fitness, complexity, path):
    conn.post_entries(model, fitness, solution, error, complexity, alg_name, iteration,
                      experiment.metrics.MSE,
                      experiment.metrics.RMSE,
                      experiment.metrics.MAE,
                      experiment.metrics.ABS_REL,
                      experiment.metrics.LOG10,
                      experiment.metrics.DELTA1,
                      experiment.metrics.DELTA2,
                      experiment.metrics.DELTA3,
                      experiment.metrics.CADL)
    torch.save(model.state_dict(), path + f"/model.pt")


class RNNVAEAEArchitecture(ExtendedProblem):

    def __init__(self, dimension):
        super().__init__(dimension=dimension, lower=0, upper=1)
        self.iteration = 0

    def _evaluate(self, solution, alg_name):
        Log.debug("=================================================================================================")
        Log.debug(f"ITERATION: {self.iteration}")
        Log.debug(f"SOLUTION : {solution}")
        self.iteration += 1

        model = RNNVAE(solution, **config)
        existing_entry = conn.get_entries(hash_id=model.hash_id)
        path = config['logging_params']['save_dir'] + str(self.iteration) + "_" + alg_name + "_" + model.hash_id
        config['logging_params']['model_path'] = path
        Path(path).mkdir(parents=True, exist_ok=True)

        if existing_entry.shape[0] > 0:
            fitness = existing_entry['fitness'][0]
            Log.info(f"Model for this solution already exists")
            return fitness

        else:
            """Punishing bad decisions"""
            if len(model.encoding_layers) == 0 or len(model.decoding_layers) == 0:
                fitness = int(9e10)
                RMSE = int(9e10)
                print(
                    f"Fitness: {fitness}, RMSE: {RMSE}, Solution: {solution}, Algorithm: {alg_name}, Iteration: {self.iteration}")
                conn.post_entries(model, fitness, solution, RMSE, alg_name, self.iteration)
            else:
                experiment = RNNVAExperiment(model, **config)
                config['trainer_params']['min_epochs'] = model.num_epochs
                config['trainer_params']['max_epochs'] = 2
                tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                              name=str(self.iteration) + "_" + alg_name + "_" + model.hash_id)

                trainer = Trainer(logger=tb_logger,
                                  enable_progress_bar=True,
                                  accelerator="cuda",
                                  devices=1,
                                  default_root_dir=tb_logger.root_dir,
                                  log_every_n_steps=50,
                                  # auto_select_gpus=True,

                                  callbacks=[
                                      LearningRateMonitor(),
                                      # BatchSizeFinder(mode="power", steps_per_trial=3),
                                      FineTuneLearningRateFinder(**config['fine_tune_lr_finder']),
                                      EarlyStopping(**config['early_stop'],
                                                    verbose=False,
                                                    check_finite=True),
                                      # ModelCheckpoint(save_top_k=1,
                                      #                 dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                      #                 monitor="loss",
                                      #                 save_last=True)
                                  ],
                                  # strategy=DDPPlugin(find_unused_parameters=False),
                                  **config['trainer_params'])

                Log.info(f"======= Training {config['model_params']['name']} =======")
                Log.info(f'\nTraining start: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                trainer.fit(experiment, datamodule=datamodule)
                Log.info(f'\nTraining end: {datetime.now().strftime("%H:%M:%S-%d/%m/%Y")}')
                trainer.test(experiment, datamodule=datamodule)

                # Known problem: https://discuss.pytorch.org/t/why-my-model-returns-nan/24329/5
                # if math.isnan(experiment.test_RMSE.item()):
                #     RMSE = int(9e10)
                # else:
                #     RMSE = experiment.test_RMSE.item()
                #
                # complexity = (model.num_epochs ** 2) + (model.num_layers * 100) + (model.bottleneck_size * 10)
                # fitness = (RMSE * 1000) + (complexity / 100)

                fitness, error, complexity = calculate_fitness(model, experiment)

                Log.debug(tabulate([[complexity, fitness]], headers=["Complexity", "Fitness"],
                                   tablefmt="pretty"))
                upload_save_model(alg_name, self.iteration, solution, error, model, experiment, fitness, complexity,
                                  path)

            if np.isnan(fitness):
                fitness = int(9e10)
            return fitness


def solve_architecture_problem(selected_algorithms):
    """
    Dimensionality:
    y1: topology shape,
    y2: layer type
    y3: number of neurons per layer,
    y4: number of layers,
    y5: activation function
    y6: number of epochs,
    y7: learning rate
    y8: optimizer algorithm.
    """
    DIMENSIONALITY = 8

    algorithms = {
        "particle_swarm": ParticleSwarmAlgorithm(),
        "differential_evolution": DifferentialEvolution(),
        "firefly_algorithm": FireflyAlgorithm(),
        "self_adaptive_differential_evolution": SelfAdaptiveDifferentialEvolution(),
        "genetic_algorithm": GeneticAlgorithm()
    }

    selected_algorithm_objects = [algorithms.get(algorithm_name) for algorithm_name in selected_algorithms if
                                  algorithms.get(algorithm_name) is not None]

    runner = ExtendedRunner(
        config['logging_params']['save_dir'],
        dimension=DIMENSIONALITY,
        optimization_type=OptimizationType.MINIMIZATION,
        max_evals=config['nia_search']['evaluations'],
        runs=config['nia_search']['runs'],
        algorithms=selected_algorithm_objects,
        problems=[
            RNNVAEAEArchitecture(DIMENSIONALITY)
        ]
    )

    Log.info("=====================================SEARCH STARTED==============================================")
    final_solutions = runner.run(export='json', verbose=True)
    Log.info("=====================================SEARCH COMPLETED============================================")

    Log.info(f"Solutions: {final_solutions}")
    best_solution, best_algorithm = conn.best_results()
    best_model = RNNVAE(best_solution, **config)
    model_file = config['logging_params']['save_dir'] + f"{best_algorithm}_{best_model.hash_id}.pt"
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference
    torch.save(best_model.state_dict(), model_file)
    Log.info(f"Best model saved to: {model_file}")
