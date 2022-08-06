import sqlite3
from datetime import datetime
import pandas as pd


class SQLiteConnector():
    def __init__(self, db_file):
        self.db_file = db_file
        self.connection = None
        self.cursor = None
        self.create_connection()
        self.create_table()
        super(SQLiteConnector, self).__init__()

    def get_entries(self, hash_id, table_name="solution", ):
        try:
            self.create_connection()
            existing_entry = pd.read_sql(f"select * from {table_name} where hash_id='{hash_id}'", self.connection)
            self.connection.close()
        except Exception as e:
            print(e)

        return existing_entry

    def post_entries(self, model, fitness, solution, RMSE, complexity, table_name="solution"):
        try:
            self.create_connection()
            df = pd.DataFrame({'hash_id': model.hash_id,
                               'timestamp': datetime.now().strftime("%H:%M %d-%m-%Y"),
                               'encoding_layers': str(model.encoding_layers),
                               'decoding_layers': str(model.decoding_layers),
                               'topology_shape': model.topology_shape,
                               'layer_type': model.layer_type,
                               'num_layers': model.num_layers,
                               'activation': str(model.activation_name),
                               'num_epochs': model.num_epochs,
                               'learning_rate': model.learning_rate,
                               'optimizer': str(model.optimizer_name),
                               'bottleneck_size': model.bottleneck_size,
                               'RMSE': RMSE,
                               'complexity': complexity,
                               'fitness': fitness,
                               'solution_array': str(solution).strip()
                               }, index=[0])
            df.to_sql(table_name, self.connection, if_exists='append', index=False)  # writes to file
            self.connection.close()
        except Exception as e:
            print(e)

    def create_table(self, table_name="solution"):
        try:
            self.cursor.execute(f'''
                       create table IF NOT EXISTS {table_name}
                        (
                            hash_id         TEXT,
                            timestamp       TEXT,
                            encoding_layers TEXT,
                            decoding_layers TEXT,
                            topology_shape  TEXT,
                            layer_type      TEXT,
                            num_layers      INTEGER,
                            activation      TEXT,
                            num_epochs      INTEGER,
                            learning_rate   REAL,
                            optimizer       TEXT,
                            bottleneck_size INTEGER,
                            RMSE            REAL,
                            complexity      INTEGER,
                            fitness         REAL,
                            solution_array  TEXT
                        );''')
            # committing our connection
            self.connection.commit()
        except Exception as e:
            print(e)

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
            print(e)
