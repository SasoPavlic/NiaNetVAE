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
# Replace NiaPy imports with pymoo
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.core.sampling import Sampling


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
        alg_name: Name of the algorithm.
        model: The model being evaluated.
        experiment: The experiment object containing metrics.
        n_features: Number of features in the dataset.
        seq_len: Sequence length of the dataset.

    Returns:
        fitness: The computed fitness value.
        error: The combined error value.
        complexity: The complexity term of the fitness function.
    """
    # Check if metrics are complete
    if not experiment.metrics.are_metrics_complete():
        Log.error("Some metric values are still None. Fitness function is waiting for metrics data.")
        return int(9e10), int(9e10), int(9e10)

    # Fetch raw metrics
    try:
        raw_metrics = experiment.metrics.compute()
        Log.debug(f"Raw metrics: {raw_metrics}")  # Debugging raw metrics
    except Exception as e:
        Log.error(f"Error computing metrics: {e}")
        return int(9e10), int(9e10), int(9e10)

    # Update database with raw metrics
    for metric_name, value in raw_metrics.items():
        Log.debug(f"Updating database for metric: {metric_name}, value: {value}")  # Debug database update
        if value != int(9e10):  # Only update valid metrics
            conn.update_min_max(dataset_name, alg_name, metric_name, value)

    for metric_name, value in experiment.anomaly_metrics.items():
        Log.debug(f"Updating database for metric: {metric_name}, value: {value}")  # Debug database update
        if value != int(9e10):  # Only update valid metrics
            conn.update_min_max(dataset_name, alg_name, metric_name, value)

    # Normalize all metrics
    normalized_metrics = {}
    for metric_name, value in raw_metrics.items():
        try:
            normalized_metrics[metric_name] = compute_normalized_metric(
                metric_name, value, False , conn, dataset_name, alg_name
            )
            Log.debug(f"Normalized metric {metric_name}: {normalized_metrics[metric_name]}")  # Debug normalized values
        except Exception as e:
            Log.error(f"Error normalizing metric {metric_name}: {e}")
            normalized_metrics[metric_name] = 1.0  # Worst normalized value


    for metric_name, value in experiment.anomaly_metrics.items():
        try:
            normalized_metrics[metric_name] = compute_normalized_metric(
                metric_name, value, False , conn, dataset_name, alg_name
            )
            Log.debug(f"Normalized metric {metric_name}: {normalized_metrics[metric_name]}")  # Debug normalized values
        except Exception as e:
            Log.error(f"Error normalizing metric {metric_name}: {e}")
            normalized_metrics[metric_name] = 1.0  # Worst normalized value



    # Ensure metrics_to_calculate is always a list
    metrics_to_calculate = config['nia_search']['metrics']
    if isinstance(metrics_to_calculate, str):
        metrics_to_calculate = [metrics_to_calculate]  # Convert a single string to a list

    Log.debug(f"Metrics to calculate: {metrics_to_calculate}")  # Debug metrics to calculate

    # Calculate error_x using metrics specified in the config
    error_x = 0.0
    for metric_name in metrics_to_calculate:
        if metric_name in normalized_metrics:
            error_x += normalized_metrics[metric_name]
        else:
            Log.error(f"Metric {metric_name} not found in normalized metrics. Available: {normalized_metrics.keys()}")

    # Complexity calculation
    def normalize_complexity(value, max_bound):
        return value / max_bound

    encoding_complexity = normalize_complexity(len(model.encoding_layers), seq_len)
    decoding_complexity = normalize_complexity(len(model.decoding_layers), seq_len)
    bottleneck_complexity = normalize_complexity(model.bottleneck_size, seq_len)

    max_possible_complexity = 3.0  # Sum of all normalized components
    complexity = int(
        round((encoding_complexity + decoding_complexity + bottleneck_complexity) / max_possible_complexity, 6) * 1000000
    )

    # Total fitness calculation
    try:
        error = int(round(error_x, 6) * 1000000)
        #error = int(round(experiment.anomaly_metrics.get('pr_auc'), 6) * 1000000)
        fitness = error + complexity
        Log.debug(f"Calculated fitness: {fitness}, error: {error}, complexity: {complexity}")  # Debug fitness values

        if math.isnan(fitness) or math.isnan(error) or math.isnan(complexity):
            Log.error("Invalid fitness, error, or complexity value detected. Returning worst possible value.")
            return int(9e10), int(9e10), int(9e10)

    except Exception as e:
        Log.error(f"Error during fitness calculation: {e}")
        return int(9e10), int(9e10), int(9e10)

    return fitness, error, complexity

class VAESearchProblem(Problem):
    def __init__(self, config, conn, datamodule, dataset_name):
        super().__init__(n_var=6, n_obj=2, n_constr=0, xl=0.0, xu=1.0)
        self.config = config
        self.conn = conn
        self.datamodule = datamodule
        self.dataset_name = dataset_name
        self.iteration = 0

    def _evaluate(self, X, out, *args, **kwargs):
        # X is population of solutions, shape (population_size, 6)
        F = []
        for solution in X:
            fitness, error, complexity = self.evaluate_solution(solution)
            F.append([error, complexity])  # Multi-objective: minimize both
        out["F"] = np.array(F)

    def evaluate_solution(self, solution):
        # Existing evaluation logic from RNNVAEAEArchitecture._evaluate
        Log.debug(f"ITERATION: {self.iteration}")
        Log.debug(f"SOLUTION : {solution}")
        self.iteration += 1

        model = RNNVAE(solution, **self.config)
        existing_entry = self.conn.get_entries(hash_id=model.get_hash(), dataset_name=self.dataset_name)
        path = self.config['logging_params']['save_dir'] + str(self.iteration) + "_pymoo_" + model.hash_id
        self.config['logging_params']['model_path'] = path
        Path(path).mkdir(parents=True, exist_ok=True)

        if not model.is_valid:
            fitness = int(9e10)
            complexity = int(9e10)
            error = int(9e10)
            self.conn.save_model_and_entry(
                dataset_name=self.dataset_name,
                alg_name="pymoo",
                iteration=self.iteration,
                model=model,
                fitness=fitness,
                solution=solution,
                error=error,
                complexity=complexity
            )
            return fitness, error, complexity

        experiment = RNNVAExperiment(model, path, self.dataset_name, "pymoo", **self.config)

        trainer = Trainer(
            enable_progress_bar=True,
            accelerator="cuda",
            devices=1,
            default_root_dir=path,
            log_every_n_steps=50,
            logger=False,
            callbacks=[
                FineTuneLearningRateFinder(**self.config['fine_tune_lr_finder']),
                EarlyStopping(**self.config['early_stop'], verbose=True, check_finite=True)
            ],
            **self.config['trainer_params']
        )

        try:
            Log.info(f"======= Training {self.config['logging_params']['name']} =======")
            start_time = datetime.now()
            trainer.fit(experiment, datamodule=self.datamodule)
            trainer.test(experiment, datamodule=self.datamodule)
            end_time = datetime.now()
        except Exception as e:
            Log.error(f"Training failed: {e}")
            return int(9e10), int(9e10), int(9e10)

        fitness, error, complexity = calculate_fitness("pymoo", model, experiment,
                                                       self.config['data_params']['n_features'],
                                                       self.config['data_params']['seq_len'])

        self.conn.save_model_and_entry(
            dataset_name=self.dataset_name,
            alg_name="pymoo",
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
            duration=(end_time - start_time).total_seconds()
        )

        return fitness, error, complexity


def solve_architecture_problem(selected_algorithms):
    # Modified to use pymoo instead of NiaPy
    problem = VAESearchProblem(
        config=config,
        conn=conn,
        datamodule=datamodule,
        dataset_name=dataset_name
    )

    algorithm = NSGA2(pop_size=config['nia_search'].get('population_size', 100))

    res = minimize(
        problem,
        algorithm,
        termination=('n_gen', config['nia_search']['evaluations']),
        seed=config['exp_params']['manual_seed'],
        verbose=True,
        save_history=True
    )

    Log.info("=====================================SEARCH COMPLETED============================================")
    Log.info(f"Pareto solutions: {res.X}")

    # Save all non-dominated solutions
    for solution in res.X:
        model = RNNVAE(solution, **config)
        model_file = config['logging_params']['save_dir'] + f"{dataset_name}_pymoo_{model.hash_id}.pt"
        torch.save(model.state_dict(), model_file)
        Log.info(f"Saved Pareto solution model to: {model_file}")

    return res.X