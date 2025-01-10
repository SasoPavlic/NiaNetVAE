import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer
# from lightning.pytorch.plugins import DDPPlugin
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from niapy.algorithms.basic import ParticleSwarmAlgorithm, DifferentialEvolution, FireflyAlgorithm, GeneticAlgorithm
from niapy.algorithms.modified import SelfAdaptiveDifferentialEvolution
from niapy.task import OptimizationType
from tabulate import tabulate

import nianetvae.experiments.metrics_evaluation
from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment, FineTuneLearningRateFinder
from nianetvae.models.rnn_vae import RNNVAE
from nianetvae.niapy_extension.wrapper import ExtendedProblem, ExtendedRunner

RUN_UUID = None
config = None
conn = None
datamodule = None
dataset_name = None


def compute_normalized_metric(metric_name, value, is_higher_better, conn, dataset_name, alg_name):
    """
    Retrieve, handle, and normalize a metric value using observed min/max values from the database.

    Args:
        metric_name (str): The name of the metric.
        value (float): The raw metric value to normalize.
        is_higher_better (bool): Whether higher values are better for this metric.
        conn: The database connection object for retrieving min/max values.
        dataset_name (str): The name of the dataset.
        alg_name (str): The name of the algorithm.

    Returns:
        float: The normalized metric value in the range [0, 1].
    """
    try:
        # Retrieve min and max values from the database
        min_val, max_val = conn.get_min_max(dataset_name, alg_name, metric_name)

        # Handle cases where no min/max values are observed yet
        if min_val == float('inf') and max_val == float('-inf'):
            Log.warning(f"No observed min/max for {metric_name}. Defaulting to value-based normalization.")
            min_val, max_val = value, value  # Use current value as both bounds
        elif min_val == float('inf'):
            Log.warning(f"No observed minimum for {metric_name}. Using current value as min.")
            min_val = value
        elif max_val == float('-inf'):
            Log.warning(f"No observed maximum for {metric_name}. Using current value as max.")
            max_val = value

        # Handle zero range
        if max_val == min_val:
            # Always return 1.0 for a zero range (worst-case normalization)
            return 1.0

        # Normalize the value
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0.0, min(1.0, normalized))  # Clamp to [0, 1]

        # Invert if higher values are better
        return 1.0 - normalized if is_higher_better else normalized

    except Exception as e:
        Log.error(f"Error normalizing metric {metric_name}: {e}")
        return 1.0  # Return worst normalized value in case of failure


def calculate_fitness(alg_name, model, experiment, n_features, seq_len):
    """
    Calculate the fitness value for the given model and experiment metrics.

    Args:
        alg_name: NIA
        model: The model being evaluated.
        experiment: The experiment object containing metrics.
        n_features: Number of features in the dataset.

    Returns:
        fitness: The computed fitness value.
        error: The combined error value.
        complexity: The complexity term of the fitness function.
    """
    if not experiment.metrics.are_metrics_complete():
        Log.error("Some metric values are still None. When fitness function waits for metrics data.")
        return int(9e10), int(9e10), int(9e10)

    # Fetch raw metrics
    raw_metrics = experiment.metrics.compute()

    # Update database with raw metric values
    conn.update_min_max(dataset_name, alg_name, "MAE", raw_metrics["MAE"])
    conn.update_min_max(dataset_name, alg_name, "MSE", raw_metrics["MSE"])
    conn.update_min_max(dataset_name, alg_name, "RMSE", raw_metrics["RMSE"])
    conn.update_min_max(dataset_name, alg_name, "R2", raw_metrics["R2"])
    if raw_metrics["DTW"] != int(9e10):  # Update DTW only if it's a valid value
        conn.update_min_max(dataset_name, alg_name, "DTW", raw_metrics["DTW"])

    # Normalize metrics using unified function
    normalized_metrics = {
        "MAE": compute_normalized_metric("MAE", raw_metrics["MAE"], False, conn, dataset_name, alg_name),
        "MSE": compute_normalized_metric("MSE", raw_metrics["MSE"], False, conn, dataset_name, alg_name),
        "RMSE": compute_normalized_metric("RMSE", raw_metrics["RMSE"], False, conn, dataset_name, alg_name),
        "R2": compute_normalized_metric("R2", raw_metrics["R2"], True, conn, dataset_name, alg_name),
        "DTW": compute_normalized_metric("DTW", raw_metrics["DTW"], False, conn, dataset_name, alg_name) if raw_metrics[
                                                                                                                "DTW"] != int(
            9e10) else 0.0,
    }

    # Calculate error_x using normalized metrics
    error_x = (
            normalized_metrics["MAE"] +
            normalized_metrics["MSE"] +
            normalized_metrics["RMSE"]
    )

    # Include DTW if applicable
    if n_features == 1:
        if "DTW" in normalized_metrics and normalized_metrics["DTW"] != int(9e10):
            error_x += normalized_metrics["DTW"]
        else:
            Log.error("DTW metric was not computed.")
    else:
        Log.error("DTW metric is not included because the dataset is not univariate.")

    # Use normalized R² directly
    error_y = normalized_metrics["R2"]

    # Complexity calculation with hardcoded bounds
    def normalize_complexity(value, max_bound):
        return value / max_bound  # Normalized to [0, 1]

    encoding_normalized_num_layers = normalize_complexity(len(model.encoding_layers), seq_len)
    decoding_normalized_num_layers = normalize_complexity(len(model.decoding_layers), seq_len)
    normalized_bottleneck = normalize_complexity(model.bottleneck_size, seq_len)

    max_possible_complexity = 3.0  # Sum of all normalized complexity components
    complexity = int(
        round((encoding_normalized_num_layers
               + decoding_normalized_num_layers
               + normalized_bottleneck)
              / max_possible_complexity,
              6)
        * 1000000
    )

    # Total fitness calculation
    try:
        error = int(round(error_x + error_y, 6) * 1000000)  # Add normalized R² to the error term
        fitness = error + complexity

        # Check for NaN or invalid values
        if math.isnan(fitness) or math.isnan(error) or math.isnan(complexity):
            Log.error("Invalid fitness, error, or complexity value detected (NaN). Setting worst possible value.")
            return int(9e10), int(9e10), int(9e10)

    except Exception as e:
        Log.error(f"Error during fitness calculation: {e}")
        return int(9e10), int(9e10), int(9e10)

    return fitness, error, complexity


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
        existing_entry = conn.get_entries(hash_id=model.get_hash(), dataset_name=dataset_name)
        path = config['logging_params']['save_dir'] + str(self.iteration) + "_" + alg_name + "_" + model.hash_id
        config['logging_params']['model_path'] = path
        Path(path).mkdir(parents=True, exist_ok=True)

        if existing_entry.shape[0] > 0 and True==False:
            fitness = existing_entry['fitness'][0]
            Log.info(f"Model for this solution already exists")
            return fitness

        else:
            """Punishing bad decisions"""
            if not model.is_valid:
                fitness = int(9e10)
                complexity = int(9e10)
                error = int(9e10)
                conn.save_model_and_entry(
                    dataset_name=dataset_name,
                    alg_name=alg_name,
                    iteration=self.iteration,
                    model=model,
                    fitness=fitness,
                    solution=solution,
                    error=error,
                    complexity=complexity
                )

            else:
                experiment = RNNVAExperiment(model, path, dataset_name, alg_name, **config)
                tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                              name=str(self.iteration) + "_" + alg_name + "_" + model.hash_id)

                trainer = Trainer(logger=tb_logger,
                                  enable_progress_bar=True,
                                  accelerator="cuda",
                                  devices=1,
                                  default_root_dir=tb_logger.root_dir,
                                  log_every_n_steps=50,
                                  # profiler="simple",
                                  # auto_select_gpus=True,

                                  callbacks=[
                                      LearningRateMonitor(),
                                      FineTuneLearningRateFinder(**config['fine_tune_lr_finder']),
                                      EarlyStopping(**config['early_stop'],
                                                    verbose=True,
                                                    check_finite=True),
                                      # BatchSizeFinder(
                                      #     mode="power",  # "power" or "binsearch" modes
                                      #     steps_per_trial=3,  # Number of steps to run with each batch size
                                      #     init_val=2,  # Initial batch size to start search with
                                      #     max_trials=25,  # Max number of trials (batch size increases) to try
                                      # ),
                                      # ModelCheckpoint(save_top_k=1,
                                      #                 dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                      #                 monitor="loss",
                                      #                 save_last=True)
                                  ],
                                  # strategy=DDPPlugin(find_unused_parameters=False),
                                  **config['trainer_params'])

                Log.info(f"======= Training {config['logging_params']['name']} =======")
                start_time = datetime.now()
                Log.info(f'\nTraining start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                trainer.fit(experiment, datamodule=datamodule)
                Log.info(f'\nTraining end: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

                Log.info(f'\nTest start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                trainer.test(experiment, datamodule=datamodule)
                Log.info(f'\nTest end: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                fitness, error, complexity = calculate_fitness(alg_name,
                                                               model,
                                                               experiment,
                                                               config['data_params']['n_features'],
                                                               config['data_params']['seq_len']
                                                               )

                Log.debug(tabulate([[complexity, fitness]], headers=["Complexity", "Fitness"],
                                   tablefmt="pretty"))
                conn.save_model_and_entry(
                    dataset_name=dataset_name,
                    alg_name=alg_name,
                    iteration=self.iteration,
                    solution=solution,
                    error=error,
                    model=model,
                    experiment=experiment,
                    fitness=fitness,
                    complexity=complexity,
                    path=path,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration
                )

            if np.isnan(fitness):
                fitness = int(9e10)
            return fitness


def solve_architecture_problem(selected_algorithms):
    """
    Dimensionality:
    y1: topology shape,
    y2: layer type
    y3: layer step,
    y4: number of layers,
    y5: activation function
    y6: optimizer algorithm.
    """
    DIMENSIONALITY = 6

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

    """Issue when using multiple GPUs
        https://github.com/Lightning-AI/pytorch-lightning/issues/2807
    """
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
