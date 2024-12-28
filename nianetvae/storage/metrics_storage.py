import pandas as pd
from log import Log
from nianetvae.storage import SQLiteBase


class ObservedMetricsDB(SQLiteBase):
    def create_table(self):
        """Create the observed_metrics table if it doesn't exist."""
        try:
            create_table_query = f'''
            CREATE TABLE IF NOT EXISTS {self.table_name} (
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
            query = f'''
            SELECT observed_min, observed_max
            FROM {self.table_name}
            WHERE dataset_name = ? AND algorithm_name = ? AND metric_name = ?
            '''
            self.cursor.execute(query, (dataset_name, algorithm_name, metric_name))
            result = self.cursor.fetchone()
            return result if result else (float('inf'), float('-inf'))
        except Exception as e:
            Log.error(f"Error retrieving min/max values: {e}")
            return float('inf'), float('-inf')

    def update_min_max(self, dataset_name, algorithm_name, metric_name, value):
        """Update min/max values dynamically."""
        try:
            current_min, current_max = self.get_min_max(dataset_name, algorithm_name, metric_name)
            new_min = min(current_min, value)
            new_max = max(current_max, value)

            query = f'''
            INSERT INTO {self.table_name} (dataset_name, algorithm_name, metric_name, observed_min, observed_max)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(dataset_name, algorithm_name, metric_name)
            DO UPDATE SET
                observed_min = excluded.observed_min,
                observed_max = excluded.observed_max
            '''
            self.cursor.execute(query, (dataset_name, algorithm_name, metric_name, new_min, new_max))
            self.connection.commit()
        except Exception as e:
            Log.error(f"Error updating min/max values: {e}")
