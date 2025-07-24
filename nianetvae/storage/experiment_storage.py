import json
import os
import random
import sqlite3
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from log import Log

# Your “infinity” penalty constant
infinity = int(9e10)


def _retry_db(max_retries=5, base_delay=0.1, jitter=0.05):
    """
    Decorator to retry a DB operation on SQLITE_BUSY with exponential backoff.
    """
    def decorator(fn):
        def wrapped(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return fn(*args, **kwargs)
                except sqlite3.OperationalError as e:
                    msg = str(e).lower()
                    if 'database is locked' in msg:
                        wait = base_delay * (2 ** attempt) + random.uniform(0, jitter)
                        Log.warning(f"DB locked in {fn.__name__}, retry {attempt + 1}/{max_retries} after {wait:.2f}s")
                        time.sleep(wait)
                        continue
                    else:
                        Log.error(f"OperationalError in {fn.__name__}: {e}")
                        raise
            Log.error(f"{fn.__name__} failed after {max_retries} retries due to SQLITE_BUSY")
        return wrapped
    return decorator


class SQLiteConnector:
    def __init__(self, db_file, table_name):
        self.db_file = db_file
        self.table_name = table_name
        # Ensure tables exist on startup, but do not fail on error
        try:
            self._create_table()
            self._create_table_metrics()
        except Exception as e:
            Log.error(f"Error initializing database tables: {e}")

    def _get_connection(self):
        """
        Open a new SQLite connection with DELETE journaling (safe on shared filesystems),
        a busy timeout, and NORMAL synchronous for performance.
        """
        conn = sqlite3.connect(
            self.db_file,
            timeout=30,  # wait up to 30s for locks
            check_same_thread=False  # allow cross-thread
        )
        cur = conn.cursor()
        # Use DELETE journaling for compatibility on HPC shared filesystems
        cur.execute("PRAGMA journal_mode=DELETE;")
        # Use NORMAL synchronous for performance/safety balance
        cur.execute("PRAGMA synchronous=NORMAL;")
        # Wait up to 5s before raising SQLITE_BUSY
        cur.execute("PRAGMA busy_timeout=5000;")
        return conn

    @_retry_db()
    def _create_table(self):
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute(f'''
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hash_id TEXT, dataset_name TEXT, algorithm_name TEXT,
                    timestamp TEXT, start_time TEXT, end_time TEXT,
                    duration INTEGER, iteration INTEGER,
                    activation TEXT, optimizer TEXT,
                    encoder_layer_step INTEGER, encoder_num_layers INTEGER,
                    decoder_num_layers INTEGER, decoder_layer_step INTEGER,
                    encoding_layers TEXT, decoding_layers TEXT,
                    bottleneck_size INTEGER,
                    fitness REAL, complexity REAL, error REAL,
                    MAE REAL, MSE REAL, RMSE REAL, MAPE REAL,
                    RMAPE REAL, SMAPE REAL,
                    precision REAL, recall REAL, f1_score REAL,
                    pr_auc REAL, pr_auc_mean REAL, pr_auc_std REAL,
                    roc_auc REAL, roc_auc_mean REAL, roc_auc_std REAL,
                    solution_array TEXT
                );
            ''')
            conn.commit()
        except Exception as e:
            Log.error(f"Error creating main table: {e}")
        finally:
            if conn:
                conn.close()

    @_retry_db()
    def _create_table_metrics(self):
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute('''
                CREATE TABLE IF NOT EXISTS observed_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_name TEXT NOT NULL,
                    algorithm_name TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    observed_min REAL NOT NULL,
                    observed_max REAL NOT NULL,
                    UNIQUE(dataset_name,algorithm_name,metric_name)
                );
            ''')
            conn.commit()
        except Exception as e:
            Log.error(f"Error creating metrics table: {e}")
        finally:
            if conn:
                conn.close()

    @_retry_db()
    def get_entries(self, hash_id, dataset_name):
        conn = None
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(
                f"SELECT * FROM {self.table_name} WHERE hash_id = ? AND dataset_name = ?",
                conn,
                params=(hash_id, dataset_name)
            )
            return df
        except Exception as e:
            Log.error(f"Error fetching entries: {e}")
            return pd.DataFrame()
        finally:
            if conn:
                conn.close()

    @_retry_db()
    def get_maximum_fitness(self):
        conn = None
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(f"SELECT MAX(fitness) AS max_fitness FROM {self.table_name}", conn)
            return df['max_fitness'][0]
        except Exception as e:
            Log.error(f"Error fetching maximum fitness: {e}")
            return None
        finally:
            if conn:
                conn.close()

    @_retry_db()
    def best_results(self):
        conn = None
        try:
            conn = self._get_connection()
            df = pd.read_sql_query(
                f"SELECT solution_array, algorithm_name, MIN(fitness) AS min_fitness FROM {self.table_name}",
                conn
            )
            sol = np.array(json.loads(df['solution_array'][0]))
            return sol, df['algorithm_name'][0]
        except Exception as e:
            Log.error(f"Error fetching best results: {e}")
            return None, None
        finally:
            if conn:
                conn.close()

    def get_min_max(self, dataset_name, algorithm_name, metric_name):
        """
        Retrieve observed min and max for a metric. Returns default infinities on error or missing.
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute('''
                SELECT observed_min, observed_max
                FROM observed_metrics
                WHERE dataset_name = ? AND algorithm_name = ? AND metric_name = ?
            ''', (dataset_name, algorithm_name, metric_name))
            row = cur.fetchone()
            if row is None:
                Log.info(f"No existing min/max for {metric_name}. Returning defaults.")
                return float('inf'), float('-inf')
            return row
        except Exception as e:
            Log.error(f"Error in get_min_max for {metric_name}: {e}")
            return float('inf'), float('-inf')
        finally:
            if conn:
                conn.close()

    def update_min_max(self, dataset_name, algorithm_name, metric_name, value):
        """
        Insert or update the min/max pair for a metric, but do not fail the script on errors.
        """
        conn = None
        try:
            current_min, current_max = self.get_min_max(dataset_name, algorithm_name, metric_name)
            new_min = min(current_min, value)
            new_max = max(current_max, value)

            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute('''
                INSERT INTO observed_metrics
                  (dataset_name, algorithm_name, metric_name, observed_min, observed_max)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_name,algorithm_name,metric_name)
                DO UPDATE SET observed_min=excluded.observed_min,
                              observed_max=excluded.observed_max;
            ''', (dataset_name, algorithm_name, metric_name, new_min, new_max))
            conn.commit()
        except Exception as e:
            Log.error(f"Failed to update min/max for {metric_name}: {e}")
        finally:
            if conn:
                conn.close()

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
        Insert only when we have a model, fitness, and solution.
        Will not fail the script on errors.
        """
        try:
            if not (model and fitness is not None and solution is not None):
                return

            anomaly = getattr(experiment, 'anomaly_metrics', {})
            mae = experiment.metrics.MAE if experiment else infinity
            mse = experiment.metrics.MSE if experiment else infinity
            rmse = experiment.metrics.RMSE if experiment else infinity
            mape = experiment.metrics.MAPE if experiment else infinity
            rmape = experiment.metrics.RMAPE if experiment else infinity
            smape = experiment.metrics.SMAPE if experiment else infinity

            self._insert_entry(
                model=model,
                fitness=fitness,
                solution=solution,
                error=error or 0,
                complexity=complexity or 0,
                dataset_name=dataset_name,
                alg_name=alg_name,
                iteration=iteration,
                mae=mae,
                mse=mse,
                rmse=rmse,
                mape=mape,
                rmape=rmape,
                smape=smape,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                precision=anomaly.get('precision'),
                recall=anomaly.get('recall'),
                f1_score=anomaly.get('f1_score'),
                pr_auc=anomaly.get('pr_auc'),
                pr_auc_mean=anomaly.get('pr_auc_mean'),
                pr_auc_std=anomaly.get('pr_auc_std'),
                roc_auc=anomaly.get('roc_auc'),
                roc_auc_mean=anomaly.get('roc_auc_mean'),
                roc_auc_std=anomaly.get('roc_auc_std'),
            )

            if model and path:
                try:
                    os.makedirs(path, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(path, "model.pt"))
                    Log.info(f"Model saved to {path}/model.pt")
                except Exception as e:
                    Log.error(f"Error saving model file: {e}")
        except Exception as e:
            Log.error(f"Error in save_model_and_entry: {e}")

    @_retry_db()
    def _insert_entry(
            self, model, fitness, solution, error, complexity,
            dataset_name, alg_name, iteration,
            mae, mse, rmse, mape, rmape, smape,
            start_time, end_time, duration,
            precision=None, recall=None, f1_score=None,
            pr_auc=None, pr_auc_mean=None, pr_auc_std=None,
            roc_auc=None, roc_auc_mean=None, roc_auc_std=None
    ):
        """
        Core insertion logic, retried on SQLITE_BUSY, but will log and continue on errors.
        """
        conn = None
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            data = {
                'hash_id': str(model.hash_id),
                'dataset_name': dataset_name,
                'algorithm_name': alg_name,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'start_time': start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None,
                'end_time': end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None,
                'duration': int(duration) if duration is not None else None,
                'iteration': int(iteration),
                'activation': model.activation_name,
                'optimizer': model.optimizer_name,
                'encoder_layer_step': int(model.encoder_layer_step),
                'encoder_num_layers': int(model.encoder_num_layers),
                'decoder_num_layers': int(model.decoder_num_layers),
                'decoder_layer_step': int(model.decoder_layer_step),
                'encoding_layers': str(model.encoding_layers),
                'decoding_layers': str(model.decoding_layers),
                'bottleneck_size': int(model.bottleneck_size),
                'fitness': float(fitness),
                'complexity': float(complexity),
                'error': float(error),
                'MAE': float(mae),
                'MSE': float(mse),
                'RMSE': float(rmse),
                'MAPE': float(mape),
                'RMAPE': float(rmape),
                'SMAPE': float(smape),
                'precision': float(precision) if precision is not None else None,
                'recall': float(recall) if recall is not None else None,
                'f1_score': float(f1_score) if f1_score is not None else None,
                'pr_auc': float(pr_auc) if pr_auc is not None else None,
                'pr_auc_mean': float(pr_auc_mean) if pr_auc_mean is not None else None,
                'pr_auc_std': float(pr_auc_std) if pr_auc_std is not None else None,
                'roc_auc': float(roc_auc) if roc_auc is not None else None,
                'roc_auc_mean': float(roc_auc_mean) if roc_auc_mean is not None else None,
                'roc_auc_std': float(roc_auc_std) if roc_auc_std is not None else None,
                'solution_array': json.dumps(solution.tolist())
            }
            cols = ','.join(data.keys())
            placeholders = ','.join('?' for _ in data)
            query = f"INSERT INTO {self.table_name} ({cols}) VALUES ({placeholders})"
            cur.execute(query, tuple(data.values()))
            conn.commit()
        except Exception as e:
            Log.error(f"Error inserting entry: {e}")
        finally:
            if conn:
                conn.close()
