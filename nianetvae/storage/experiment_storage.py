import json
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from log import Log

infinity = int(9e10)


class SQLiteConnector:
    def __init__(self, db_file, table_name):
        self.db_file = db_file
        self.table_name = table_name
        self.connection = None
        self.cursor = None
        self.create_connection()
        self.create_table()
        self.create_table_metrics()

    def create_connection(self):
        """Create a database connection to the SQLite database specified by the db_file."""
        try:
            self.connection = sqlite3.connect(self.db_file, timeout=10)
            self.cursor = self.connection.cursor()
        except Exception as e:
            Log.error(f"Error creating database connection: {e}")

    def create_table(self):
        """Create the solutions table if it doesn't exist, including anomaly detection metrics."""
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
                MAPE REAL,
                RMAPE REAL,
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
            self.create_connection()
            query = f"SELECT * FROM {self.table_name} WHERE hash_id = ?"
            existing_entry = pd.read_sql_query(query, self.connection, params=(hash_id,))
            query = f"SELECT * FROM {self.table_name} WHERE hash_id = ? AND dataset_name = ?"
            existing_entry = pd.read_sql_query(query, self.connection, params=(hash_id, dataset_name))
            self.connection.close()
        except Exception as e:
            Log.error(f"Could not get existing entries: {e}")
            existing_entry = pd.DataFrame()
        return existing_entry

    def get_maximum_fitness(self):
        """Retrieve the maximum fitness value from the database."""
        try:
            self.create_connection()
            query = f"SELECT MAX(fitness) AS max_fitness FROM {self.table_name}"
            maximum_results = pd.read_sql_query(query, self.connection)
            self.connection.close()
            max_fitness = maximum_results['max_fitness'][0]
            return max_fitness
        except Exception as e:
            Log.error(f"Error getting maximum fitness: {e}")
            return None

    def best_results(self):
        """Retrieve the best solution (with the minimum fitness) from the database."""
        try:
            self.create_connection()
            query = f"SELECT solution_array, algorithm_name, MIN(fitness) AS min_fitness FROM {self.table_name}"
            best_results = pd.read_sql_query(query, self.connection)
            self.connection.close()

            best_solution_json = best_results['solution_array'][0]
            best_solution = np.array(json.loads(best_solution_json))
            best_algorithm = best_results['algorithm_name'][0]

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

            # Insert entry into the database
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
                    mae=experiment.metrics.MAE if experiment else infinity,
                    mse=experiment.metrics.MSE if experiment else infinity,
                    rmse=experiment.metrics.RMSE if experiment else infinity,
                    mape=experiment.metrics.MAPE if experiment else infinity,
                    rmape=experiment.metrics.RMAPE if experiment else infinity,
                    dtw=experiment.metrics.DTW if experiment else infinity,
                    r2=experiment.metrics.R2 if experiment else float('-inf'),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    **metrics,
                )

            # Save the model state
            if model and path:
                model_path = os.path.join(path, "model.pt")
                torch.save(model.state_dict(), model_path)
                Log.info(f"Model saved to {model_path}")

        except Exception as e:
            Log.error(f"Error saving model and entry: {e}")

    def _insert_entry(self, model, fitness, solution, error, complexity, dataset_name, alg_name, iteration,
                      mae, mse, rmse, mape, rmape, dtw, r2, start_time, end_time, duration,
                      precision=None, recall=None, f1_score=None,
                      pr_auc=None, pr_auc_mean=None, pr_auc_std=None,
                      roc_auc=None, roc_auc_mean=None, roc_auc_std=None):
        """Insert a new entry into the database."""
        try:
            self.create_connection()
            json_solution = json.dumps(solution.tolist())
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
                'MAPE': float(mape),
                'RMAPE': float(rmape),
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

            # Insert into the table
            columns = ', '.join(data.keys())
            placeholders = ', '.join(['?' for _ in data])
            query = f"INSERT INTO {self.table_name} ({columns}) VALUES ({placeholders})"
            self.cursor.execute(query, tuple(data.values()))
            self.connection.commit()
            self.connection.close()
        except Exception as e:
            Log.error(f"Error inserting entry: {e}")


    def create_table_metrics(self):
        """Create the observed_metrics table if it doesn't exist."""
        try:
            create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {"observed_metrics"} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT NOT NULL,
                algorithm_name TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                observed_min REAL NOT NULL,
                observed_max REAL NOT NULL,
                UNIQUE(dataset_name, algorithm_name, metric_name)
            );
            '''
            self.cursor.execute(create_table_query)
            self.connection.commit()
        except Exception as e:
            Log.error(f"Error creating table: {e}")

    def get_min_max(self, dataset_name, algorithm_name, metric_name):
        """Retrieve current min and max values for a specific metric."""
        try:
            self.create_connection()
            query = f'''
            SELECT observed_min, observed_max
            FROM {"observed_metrics"}
            WHERE dataset_name = ? AND algorithm_name = ? AND metric_name = ?
            '''
            self.cursor.execute(query, (dataset_name, algorithm_name, metric_name))
            result = self.cursor.fetchone()
            self.connection.close()

            # If no result found, return infinities
            if result is None:
                Log.info(f"No existing min/max for {metric_name}. Returning defaults.")
                return float('inf'), float('-inf')
            else:
                return result

        except Exception as e:
            Log.error(f"Error retrieving min/max values: {e}")
            return float('inf'), float('-inf')

    def update_min_max(self, dataset_name, algorithm_name, metric_name, value):
        """Update min/max values dynamically."""
        try:
            # Retrieve current min and max values
            #TODO Which infinity goes where (min, max)
            current_min, current_max = self.get_min_max(dataset_name, algorithm_name, metric_name)
            self.create_connection()
            new_min = min(current_min, value)
            new_max = max(current_max, value)

            # Insert or update the min and max values
            query = f'''
            INSERT INTO {"observed_metrics"} (dataset_name, algorithm_name, metric_name, observed_min, observed_max)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_name, algorithm_name, metric_name)
            DO UPDATE SET
                observed_min = excluded.observed_min,
                observed_max = excluded.observed_max
            '''
            self.cursor.execute(query, (dataset_name, algorithm_name, metric_name, new_min, new_max))
            self.connection.commit()
            self.connection.close()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                Log.warning(f"Database is locked when updating min/max values: {e}")
            else:
                Log.error(f"Error updating min/max values: {e}")
        except Exception as e:
            Log.error(f"Unexpected error updating min/max values: {e}")
