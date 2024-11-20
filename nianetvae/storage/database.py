# database.py

import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

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

    def create_connection(self):
        """Create a database connection to the SQLite database specified by the db_file."""
        try:
            self.connection = sqlite3.connect(self.db_file)
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

    def get_entries(self, hash_id):
        """Retrieve entries from the database matching the given hash_id."""
        try:
            self.create_connection()
            query = f"SELECT * FROM {self.table_name} WHERE hash_id = ?"
            existing_entry = pd.read_sql_query(query, self.connection, params=(hash_id,))
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

    def post_entries(self, model, fitness, solution, error, complexity, dataset_name, alg_name, iteration,
                     mse=infinity, rmse=infinity, mae=infinity, dtw=infinity, r2=float('-inf'),
                     start_time=None, end_time=None, duration=None,
                     precision=None, recall=None, f1_score=None,
                     pr_auc=None,
                     pr_auc_mean=None,
                     pr_auc_std=None,
                     roc_auc=None,
                     roc_auc_mean=None,
                     roc_auc_std=None
                     ):
        """Insert a new entry into the database, including anomaly detection metrics."""
        try:
            self.create_connection()
            json_solution = json.dumps(solution.tolist())

            # Convert timestamps to strings
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None

            # Prepare the data as a dictionary
            data = {
                'hash_id': str(model.hash_id),
                'dataset_name' : str(dataset_name),
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
            self.connection.close()
        except Exception as e:
            Log.error(f"Error posting entries: {e}")
