import json
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from log import Log

infinity = float(99999999999)


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
                     MSE=infinity,
                     RMSE=infinity,
                     MAE=infinity,
                     ABS_REL=infinity,
                     LOG10=infinity,
                     DELTA1=infinity,
                     DELTA2=infinity,
                     DELTA3=infinity,
                     CADL=infinity):
        try:
            self.create_connection()
            json_solution = json.dumps(solution.tolist())

            df = pd.DataFrame({'hash_id': str(model.hash_id),
                               'timestamp': str(datetime.now().strftime("%H:%M %d-%m-%Y")),
                               'algorithm_name': str(alg_name),
                               'iteration': int(iteration),
                               'encoding_layers': str(model.encoding_layers),
                               'decoding_layers': str(model.decoding_layers),
                               'num_layers': int(model.num_layers),
                               'activation': str(model.activation_name),
                               'optimizer': str(model.optimizer_name),
                               'bottleneck_size': int(model.bottleneck_size),
                               'complexity': int(complexity),
                               'error': float(error),
                               'MSE': float(MSE),
                               'RMSE': float(RMSE),
                               'MAE': float(MAE),
                               'ABS_REL': float(ABS_REL),
                               'LOG10': float(LOG10),
                               'DELTA1': float(DELTA1),
                               'DELTA2': float(DELTA2),
                               'DELTA3': float(DELTA3),
                               'CADL': float(CADL),
                               'fitness': int(fitness),
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
                            timestamp       TEXT,
                            algorithm_name  TEXT,
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
                            MSE            REAL,
                            RMSE            REAL,
                            MAE            REAL,
                            ABS_REL            REAL,
                            LOG10            REAL,
                            DELTA1            REAL,
                            DELTA2            REAL,
                            DELTA3            REAL,
                            CADL            REAL,
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
