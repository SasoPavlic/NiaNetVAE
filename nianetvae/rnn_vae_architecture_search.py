import math
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from tabulate import tabulate

from log import Log
from nianetvae.experiments.rnn_vae_experiment import RNNVAExperiment, FineTuneLearningRateFinder
from nianetvae.models.rnn_vae import RNNVAE

# Global variables that are set by main.py before calling solve_architecture_problem
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
        fitness: The computed combined fitness value (error + complexity).
        error: The computed error term.
        complexity: The computed complexity term.
    """
    # Check if metrics are complete
    if not experiment.metrics.are_metrics_complete():
        Log.error("Some metric values are still None. Fitness function is waiting for metrics data.")
        return int(9e10), int(9e10), int(9e10)

    # Fetch raw metrics
    try:
        raw_metrics = experiment.metrics.compute()
        Log.debug(f"Raw metrics: {raw_metrics}")
    except Exception as e:
        Log.error(f"Error computing metrics: {e}")
        return int(9e10), int(9e10), int(9e10)

    # Update database with raw metrics (skip None, NaN or infinity‐sentinel)
    for metric_name, value in raw_metrics.items():
        Log.debug(f"Updating database for metric: {metric_name}, value: {value}")
        if value is None or (isinstance(value, float) and math.isnan(value)) or value == int(9e10):
            Log.warning(f"Skipping observed‐min/max update for raw metric '{metric_name}': invalid value {value}")
            continue
        conn.update_min_max(dataset_name, alg_name, metric_name, value)

    anomaly_metrics = getattr(experiment, "anomaly_metrics", None)
    if isinstance(anomaly_metrics, dict) and anomaly_metrics:
        # Update database with anomaly metrics (skip None, NaN or infinity‐sentinel)
        for metric_name, value in anomaly_metrics.items():
            Log.debug(f"Updating database for anomaly metric: {metric_name}, value: {value}")
            if value is None or (isinstance(value, float) and math.isnan(value)) or value == int(9e10):
                Log.warning(
                    f"Skipping observed‐min/max update for anomaly metric '{metric_name}': invalid value {value}"
                )
                continue
            conn.update_min_max(dataset_name, alg_name, metric_name, value)

    # Normalize all metrics
    normalized_metrics = {}
    for metric_name, value in raw_metrics.items():
        try:
            normalized_metrics[metric_name] = compute_normalized_metric(
                metric_name, value, False, conn, dataset_name, alg_name
            )
            Log.debug(f"Normalized metric {metric_name}: {normalized_metrics[metric_name]}")
        except Exception as e:
            Log.error(f"Error normalizing metric {metric_name}: {e}")
            normalized_metrics[metric_name] = 1.0

    if isinstance(anomaly_metrics, dict) and anomaly_metrics:
        for metric_name, value in anomaly_metrics.items():
            try:
                normalized_metrics[metric_name] = compute_normalized_metric(
                    metric_name, value, False, conn, dataset_name, alg_name
                )
                Log.debug(f"Normalized metric {metric_name}: {normalized_metrics[metric_name]}")
            except Exception as e:
                Log.error(f"Error normalizing metric {metric_name}: {e}")
                normalized_metrics[metric_name] = 1.0

    # Ensure metrics_to_calculate is always a list
    metrics_to_calculate = config['nia_search']['metrics']
    if isinstance(metrics_to_calculate, str):
        metrics_to_calculate = [metrics_to_calculate]

    Log.debug(f"Metrics to calculate: {metrics_to_calculate}")

    # Calculate error_x using metrics specified in the config.
    # Also check that at least one metric was found and added.
    error_x = 0.0
    found_any = False
    for metric_name in metrics_to_calculate:
        if metric_name in normalized_metrics:
            error_x += normalized_metrics[metric_name]
            found_any = True
        else:
            Log.error(
                f"Metric {metric_name} not found in normalized metrics. Available: {list(normalized_metrics.keys())}"
            )

    # If no metrics were found or error_x remains zero, return worst possible value.
    if not found_any or error_x == 0.0:
        Log.error("No valid metric was added to error_x or error_x remains zero. Returning worst possible value.")
        return int(9e10), int(9e10), int(9e10)

    # Complexity calculation
    def normalize_complexity(value, max_bound):
        return value / max_bound

    encoding_complexity = normalize_complexity(len(model.encoding_layers), seq_len)
    decoding_complexity = normalize_complexity(len(model.decoding_layers), seq_len)
    bottleneck_complexity = normalize_complexity(model.bottleneck_size, seq_len)

    max_possible_complexity = 3.0  # Sum of all normalized components
    complexity = int(
        round((encoding_complexity + decoding_complexity + bottleneck_complexity) / max_possible_complexity, 6)
        * 1000000
    )

    # Total fitness calculation
    try:
        error = int(round(error_x, 6) * 1000000)
        fitness = error + complexity
        Log.debug(f"Calculated fitness: {fitness}, error: {error}, complexity: {complexity}")

        if math.isnan(fitness) or math.isnan(error) or math.isnan(complexity):
            Log.error("Invalid fitness, error, or complexity value detected. Returning worst possible value.")
            return int(9e10), int(9e10), int(9e10)

    except Exception as e:
        Log.error(f"Error during fitness calculation: {e}")
        return int(9e10), int(9e10), int(9e10)

    return fitness, error, complexity


class RNNVAEArchitectureMultiObj(Problem):
    """
    This class defines the multiobjective problem for RNN-VAE architecture search.
    The two objectives are:
      1. The error (from validation/test metrics)
      2. The model complexity (based on the number of layers and bottleneck size)
    Both are to be minimized.
    """

    def __init__(self, dimension, config, conn, datamodule, dataset_name):
        self.config = config
        self.conn = conn
        self.datamodule = datamodule
        self.dataset_name = dataset_name
        self.iteration = 0
        super().__init__(n_var=dimension, n_obj=2, n_constr=0, xl=0, xu=1)

    def _evaluate(self, X, out, *args, **kwargs):
        # X is an array of candidate solutions, shape (n_individuals, dimension)
        F = []
        for solution in X:
            self.iteration += 1
            Log.debug("=" * 100)
            Log.debug(f"ITERATION: {self.iteration}")
            Log.debug(f"SOLUTION : {solution}")

            model = RNNVAE(solution, **self.config)
            existing_entry = self.conn.get_entries(hash_id=model.get_hash(), dataset_name=self.dataset_name)

            path = self.config['logging_params']['save_dir']
            Path(path).mkdir(parents=True, exist_ok=True)

            if existing_entry.shape[0] > 0:
                error = existing_entry['error'][0]
                complexity = existing_entry['complexity'][0]
            else:
                # If the model configuration is invalid, assign worst values.
                if not model.is_valid:
                    error = int(9e10)
                    complexity = int(9e10)
                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA2",
                        iteration=self.iteration,
                        model=model,
                        fitness=int(9e10),
                        solution=solution,
                        error=error,
                        complexity=complexity
                    )
                else:
                    experiment = RNNVAExperiment(model, self.dataset_name, "NSGA2", **self.config)
                    trainer = Trainer(
                        enable_progress_bar=True,
                        accelerator="cuda",
                        devices=1,
                        default_root_dir=path,
                        log_every_n_steps=50,
                        logger=False,
                        enable_checkpointing=False,
                        callbacks=[
                            FineTuneLearningRateFinder(**self.config['fine_tune_lr_finder']),
                            EarlyStopping(**self.config['early_stop'], verbose=True, check_finite=True),
                        ],
                        **self.config['trainer_params']
                    )

                    Log.info(f"======= Training {self.config['logging_params']['name']} =======")
                    start_time = datetime.now()
                    Log.info(f'\nTraining start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    trainer.fit(experiment, datamodule=self.datamodule)
                    Log.info(f'\nTraining end: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    Log.info(f'\nTest start: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    trainer.test(experiment, datamodule=self.datamodule)
                    Log.info(f'\nTest end: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
                    end_time = datetime.now()
                    duration = (end_time - start_time).total_seconds()

                    fitness, error, complexity = calculate_fitness(
                        "NSGA2",
                        model,
                        experiment,
                        self.config['data_params']['n_features'],
                        self.config['data_params']['seq_len']
                    )

                    Log.debug(tabulate([[fitness, complexity, error]], headers=["Fitness, Complexity", "Error"],
                                       tablefmt="pretty"))
                    self.conn.save_model_and_entry(
                        dataset_name=self.dataset_name,
                        alg_name="NSGA2",
                        iteration=self.iteration,
                        solution=solution,
                        error=error,
                        model=model,
                        experiment=experiment,
                        fitness=fitness,
                        complexity=complexity,
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration
                    )

            # Ensure that if any value is nan, we use the worst-case penalty.
            if np.isnan(error) or np.isnan(complexity):
                error = int(9e10)
                complexity = int(9e10)
            F.append([error, complexity])
        out["F"] = np.array(F)


def solve_architecture_problem(selected_algorithms):
    """
    Uses pymoo's NSGA-II to perform a multiobjective search for the optimal balance between
    error and complexity. Optimizer settings (e.g., population size, termination criteria) are
    taken from the configuration file.
    """
    DIMENSIONALITY = 7

    # Create the multiobjective problem instance.
    # RNNVAEArchitectureMultiObj is assumed to be defined elsewhere.
    problem = RNNVAEArchitectureMultiObj(
        dimension=DIMENSIONALITY,
        config=config,
        conn=conn,
        datamodule=datamodule,
        dataset_name=dataset_name
    )

    # Determine termination criteria.
    # Use time-based termination if a time limit is provided in the config

    # Expected format: "HH:MM:SS" (e.g., "95:00:00")
    time_str = config['nia_search']['time']
    try:
        hours, minutes, seconds = map(int, time_str.split(":"))
        max_time = hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        Log.error(f"Error parsing time limit from config: {time_str}. Ensure it is in HH:MM:SS format.")
        raise e
    termination = get_termination("time", max_time=max_time)
    Log.info(f"Using time-based termination: {time_str} (={max_time} seconds)")

    # Set up the NSGA-II algorithm with the population size from config.
    algorithm = NSGA2(pop_size=config['nia_search']['population_size'])

    # Determine the number of parallel jobs (using CUDA if available).
    n_jobs = torch.cuda.device_count() if torch.cuda.is_available() else 1

    Log.info("=====================================SEARCH STARTED==============================================")
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=config['exp_params']['manual_seed'],
        verbose=True,
        n_jobs=n_jobs
    )
    Log.info("=====================================SEARCH COMPLETED============================================")
    Log.info(f"Solutions: {res.X}")

    # Retrieve the best solution from the database (using your existing criteria).
    best_solution, best_algorithm = conn.best_results(dataset_name)
    best_model = RNNVAE(best_solution, **config)
    model_file = config['logging_params']['save_dir'] + f"{dataset_name}_NSGA2_{best_model.hash_id}.pt"
    torch.save(best_model.state_dict(), model_file)
    Log.info(f"Best model saved to: {model_file}")
