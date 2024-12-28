import sqlite3
import pandas as pd
from log import Log


class ObservedMetricsDB:
    def __init__(self, db_file, table_name):
        self.db_file = db_file
        self.table_name = table_name
        self.create_table()

    def create_table(self):
        """Create the observed_metrics table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_file, timeout=10) as conn:
                cursor = conn.cursor()
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
                cursor.execute(create_table_query)
                conn.commit()
        except Exception as e:
            Log.error(f"Error creating table: {e}")

    def get_min_max(self, dataset_name, algorithm_name, metric_name):
        """Retrieve current min and max values for a specific metric."""
        try:
            with sqlite3.connect(self.db_file, timeout=10) as conn:
                cursor = conn.cursor()
                query = f'''
                SELECT observed_min, observed_max
                FROM {self.table_name}
                WHERE dataset_name = ? AND algorithm_name = ? AND metric_name = ?
                '''
                cursor.execute(query, (dataset_name, algorithm_name, metric_name))
                result = cursor.fetchone()
                return result if result else (float('inf'), float('-inf'))
        except Exception as e:
            Log.error(f"Error retrieving min/max values: {e}")
            return float('inf'), float('-inf')

    def update_min_max(self, dataset_name, algorithm_name, metric_name, value):
        """Update min/max values dynamically."""
        try:
            with sqlite3.connect(self.db_file, timeout=10) as conn:
                cursor = conn.cursor()

                # Retrieve current min and max values
                current_min, current_max = self.get_min_max(dataset_name, algorithm_name, metric_name)
                new_min = min(current_min, value)
                new_max = max(current_max, value)

                # Insert or update the min and max values
                query = f'''
                INSERT INTO {self.table_name} (dataset_name, algorithm_name, metric_name, observed_min, observed_max)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(dataset_name, algorithm_name, metric_name)
                DO UPDATE SET
                    observed_min = excluded.observed_min,
                    observed_max = excluded.observed_max
                '''
                cursor.execute(query, (dataset_name, algorithm_name, metric_name, new_min, new_max))
                conn.commit()
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e):
                Log.warning(f"Database is locked when updating min/max values: {e}")
            else:
                Log.error(f"Error updating min/max values: {e}")
        except Exception as e:
            Log.error(f"Unexpected error updating min/max values: {e}")
