import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from log import Log
from nianetvae.storage import SQLiteBase

infinity = int(9e10)


class SQLiteConnector(SQLiteBase):
    def create_table(self):
        """Create the solutions table if it doesn't exist."""
        try:
            create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hash_id TEXT,
                dataset_name TEXT,
                algorithm_name TEXT,
                timestamp TEXT,
                start_time TEXT,
                end_time TEXT,
                duration INTEGER,
                iteration INTEGER,
                activation TEXT,
                optimizer TEXT,
                shape TEXT,
                encoding_layers TEXT,
                decoding_layers TEXT,
                num_layers INTEGER,
                bottleneck_size INTEGER,
                fitness REAL,
                complexity REAL,
                error REAL,
                MAE REAL,
                MSE REAL,
                RMSE REAL,
                DTW REAL,
                R2 REAL,
                precision REAL,
                recall REAL,
                f1_score REAL,
                pr_auc REAL,
                pr_auc_mean REAL,
                pr_auc_std REAL,
                roc_auc REAL,
                roc_auc_mean REAL,
                roc_auc_std REAL,                
                solution_array TEXT
            );
            '''
            self.cursor.execute(create_table_query)
            self.connection.commit()
        except Exception as e:
            Log.error(f"Error creating table: {e}")

    def get_entries(self, hash_id, dataset_name):
        """Retrieve entries from the database matching the given hash_id and dataset_name."""
        try:
            query = f"SELECT * FROM {self.table_name} WHERE hash_id = ? AND dataset_name = ?"
            return pd.read_sql_query(query, self.connection, params=(hash_id, dataset_name))
        except Exception as e:
            Log.error(f"Could not get existing entries: {e}")
            return pd.DataFrame()

    def get_maximum_fitness(self):
        """Retrieve the maximum fitness value from the database."""
        try:
            query = f"SELECT MAX(fitness) AS max_fitness FROM {self.table_name}"
            max_fitness = pd.read_sql_query(query, self.connection).iloc[0]['max_fitness']
            return max_fitness
        except Exception as e:
            Log.error(f"Error getting maximum fitness: {e}")
            return None

    def best_results(self):
        """Retrieve the best solution (with the minimum fitness) from the database."""
        try:
            query = f"SELECT solution_array, algorithm_name, MIN(fitness) AS min_fitness FROM {self.table_name}"
            best_results = pd.read_sql_query(query, self.connection)
            best_solution = np.array(json.loads(best_results.iloc[0]['solution_array']))
            best_algorithm = best_results.iloc[0]['algorithm_name']
            return best_solution, best_algorithm
        except Exception as e:
            Log.error(f"Error getting best results: {e}")
            return None, None

    def save_model_and_entry(
            self,
            dataset_name,
            alg_name,
            iteration,
            solution=None,
            error=None,
            model=None,
            experiment=None,
            fitness=None,
            complexity=None,
            path=None,
            start_time=None,
            end_time=None,
            duration=None,
    ):
        """
        Save a model state and/or insert a new entry into the database.

        Args:
            dataset_name (str): Name of the dataset.
            alg_name (str): Name of the algorithm.
            iteration (int): Iteration number.
            solution (Optional[np.ndarray]): The solution array.
            error (Optional[float]): The error value.
            model (Optional[torch.nn.Module]): The model to save and log.
            experiment (Optional[object]): The experiment containing metrics.
            fitness (Optional[float]): The fitness value.
            complexity (Optional[float]): The complexity value.
            path (Optional[str]): Path to save the model state.
            start_time (Optional[datetime]): Training start time.
            end_time (Optional[datetime]): Training end time.
            duration (Optional[float]): Training duration in seconds.
        """
        try:
            # Extract metrics if experiment is provided
            anomaly_metrics = getattr(experiment, 'anomaly_metrics', {})
            metrics = {
                'precision': anomaly_metrics.get('precision'),
                'recall': anomaly_metrics.get('recall'),
                'f1_score': anomaly_metrics.get('f1_score'),
                'pr_auc': anomaly_metrics.get('pr_auc'),
                'pr_auc_mean': anomaly_metrics.get('pr_auc_mean'),
                'pr_auc_std': anomaly_metrics.get('pr_auc_std'),
                'roc_auc': anomaly_metrics.get('roc_auc'),
                'roc_auc_mean': anomaly_metrics.get('roc_auc_mean'),
                'roc_auc_std': anomaly_metrics.get('roc_auc_std'),
            }

            # Insert entry into the database if sufficient information is provided
            if model and fitness is not None and solution is not None:
                self._insert_entry(
                    model=model,
                    fitness=fitness,
                    solution=solution,
                    error=error or 0,
                    complexity=complexity or 0,
                    dataset_name=dataset_name,
                    alg_name=alg_name,
                    iteration=iteration,
                    mse=experiment.metrics.MSE if experiment else infinity,
                    rmse=experiment.metrics.RMSE if experiment else infinity,
                    mae=experiment.metrics.MAE if experiment else infinity,
                    dtw=experiment.metrics.DTW if experiment else infinity,
                    r2=experiment.metrics.R2 if experiment else float('-inf'),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    **metrics,
                )

            # Save the model state if a path is provided
            if model and path:
                model_path = os.path.join(path, "model.pt")
                torch.save(model.state_dict(), model_path)
                Log.info(f"Model saved to {model_path}")

        except Exception as e:
            Log.error(f"Error saving model and entry: {e}")

    def _insert_entry(self, model, fitness, solution, error, complexity, dataset_name, alg_name, iteration,
                      mse, rmse, mae, dtw, r2, start_time, end_time, duration,
                      precision=None, recall=None, f1_score=None,
                      pr_auc=None, pr_auc_mean=None, pr_auc_std=None,
                      roc_auc=None, roc_auc_mean=None, roc_auc_std=None):
        """Insert a new entry into the database."""
        try:
            json_solution = json.dumps(solution.tolist())

            # Convert timestamps to strings
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None

            # Prepare the data as a dictionary
            data = {
                'hash_id': str(model.hash_id),
                'dataset_name': str(dataset_name),
                'algorithm_name': str(alg_name),
                'timestamp': timestamp,
                'start_time': start_time_str,
                'end_time': end_time_str,
                'duration': int(duration) if duration is not None else None,
                'iteration': int(iteration),
                'activation': str(model.activation_name),
                'optimizer': str(model.optimizer_name),
                'shape': str(model.shape),
                'encoding_layers': str(model.encoding_layers) if model.encoding_layers is not None else None,
                'decoding_layers': str(model.decoding_layers) if model.decoding_layers is not None else None,
                'num_layers': int(model.num_layers),
                'bottleneck_size': int(model.bottleneck_size) if model.bottleneck_size is not None else None,
                'fitness': float(fitness),
                'complexity': float(complexity),
                'error': float(error),
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse),
                'DTW': float(dtw),
                'R2': float(r2),
                'precision': float(precision) if precision is not None else None,
                'recall': float(recall) if recall is not None else None,
                'f1_score': float(f1_score) if f1_score is not None else None,
                'pr_auc': float(pr_auc) if pr_auc is not None else None,
                'pr_auc_mean': float(pr_auc_mean) if pr_auc_mean is not None else None,
                'pr_auc_std': float(pr_auc_std) if pr_auc_std is not None else None,
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'roc_auc_mean': float(roc_auc_mean) if roc_auc_mean is not None else None,
                'roc_auc_std': float(roc_auc_std) if roc_auc_std is not None else None,
                'solution_array': json_solution.strip()
            }

            # Prepare column names and placeholders for SQL query
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])

            insert_query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
            self.cursor.execute(insert_query, tuple(data.values()))
            self.connection.commit()
        except Exception as e:
            Log.error(f"Error inserting entry: {e}")
