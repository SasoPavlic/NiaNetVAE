import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from log import Log

infinity = int(9e10)


class SQLiteConnector():
    def __init__(self, db_file, table_name):
        self.db_file = db_file
        self.table_name = table_name
        self.connection = None
        self.cursor = None
        self.create_connection()
        self.create_table()
        super(SQLiteConnector, self).__init__()

    def get_entries(self, hash_id):
        try:
            self.create_connection()
            existing_entry = pd.read_sql(f"select * from {self.table_name} where hash_id='{hash_id}'", self.connection)
            self.connection.close()
        except Exception as e:
            Log.error(f"Could not get existing entries:\n {e}")
            existing_entry = pd.DataFrame()

        return existing_entry

    def get_maximum_fitness(self):
        try:
            self.create_connection()
            maximum_results = pd.read_sql(f"select max(fitness) from {self.table_name}", self.connection)
            self.connection.close()
        except Exception as e:
            Log.error(e)

        max_fitness = maximum_results['fitness'][0]
        return max_fitness

    def best_results(self):
        try:
            self.create_connection()
            best_results = pd.read_sql(f"select solution_array, algorithm_name, min(fitness) from '{self.table_name}'",
                                       self.connection)
            self.connection.close()
        except Exception as e:
            Log.error(e)

        best_solution_json = best_results['solution_array'][0]
        best_solution = np.array(json.loads(best_solution_json))
        best_algorithm = best_results['algorithm_name'][0]

        return best_solution, best_algorithm

    def post_entries(self, model, fitness, solution, error, complexity, alg_name, iteration,
                     MAE=infinity, MSE=infinity, RMSE=infinity, DTW=infinity, R2=infinity,
                     start_time=None, end_time=None, duration=None):
        try:
            self.create_connection()
            json_solution = json.dumps(solution.tolist())

            # Convert timestamp, start_time and end_time to strings when storing in the database
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else None
            end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S") if end_time else None

            df = pd.DataFrame({'hash_id': str(model.hash_id),
                               'algorithm_name': str(alg_name),
                               'timestamp': timestamp,
                               'start_time': start_time_str,  # Store the start time as a string
                               'end_time': end_time_str,  # Store the end time as a string
                               'duration': int(duration),  # Store the duration
                               'iteration': int(iteration),
                               'encoding_layers': str(model.encoding_layers),
                               'decoding_layers': str(model.decoding_layers),
                               'num_layers': int(model.num_layers),
                               'activation': str(model.activation_name),
                               'optimizer': str(model.optimizer_name),
                               'bottleneck_size': int(model.bottleneck_size),
                               'fitness': int(fitness),
                               'complexity': int(complexity),
                               'error': float(error),
                               'MAE': float(MAE),
                               'MSE': float(MSE),
                               'RMSE': float(RMSE),
                               'DTW': float(DTW),
                               'R2': float(R2),
                               'solution_array': str(json_solution).strip()
                               }, index=[0])
            df.to_sql(self.table_name, self.connection, if_exists='append', index=False)  # writes to file
            self.connection.close()
        except Exception as e:
            Log.error(e)

    def create_table(self):
        try:
            self.cursor.execute(f'''
                       create table IF NOT EXISTS {self.table_name}
                        (
                            hash_id         TEXT,
                            algorithm_name  TEXT,
                            timestamp       TEXT,
                            start_time      TEXT,  # Add a start time column
                            end_time        TEXT,  # Add an end time column
                            duration(s)        INTEGER,  # Add a duration column
                            iteration       INTEGER,
                            activation      TEXT,
                            optimizer       TEXT,
                            encoding_layers TEXT,
                            decoding_layers TEXT,
                            num_layers      INTEGER,
                            bottleneck_size INTEGER,
                            fitness         INTEGER,
                            complexity      INTEGER,
                            error           REAL,
                            MAE             REAL,
                            MSE             REAL,
                            RMSE            REAL,
                            DTW             REAL,
                            R2              REAL,
                            solution_array  TEXT
                        );''')
            # committing our connection
            self.connection.commit()
        except Exception as e:
            Log.error(e)

    def create_connection(self):
        """ create a database connection to the SQLite database
            specified by the db_file
        :param db_file: database file
        :return: Connection object or None
        """

        try:
            self.connection = sqlite3.connect(self.db_file)
            self.cursor = self.connection.cursor()
            # create a cursor object from the cursor class
            # close our connection

        except Exception as e:
            Log.error(e)
