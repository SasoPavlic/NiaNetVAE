import sqlite3
from log import Log


class SQLiteBase:
    def __init__(self, db_file, table_name):
        self.db_file = db_file
        self.table_name = table_name
        self.connection = None
        self.cursor = None
        self.create_connection()
        self.create_table()

    def create_connection(self):
        """Create a database connection to the SQLite database specified by db_file."""
        try:
            self.connection = sqlite3.connect(self.db_file)
            self.cursor = self.connection.cursor()
        except Exception as e:
            Log.error(f"Error creating database connection: {e}")

    def create_table(self):
        """Abstract method for table creation. Must be implemented by child classes."""
        raise NotImplementedError("Subclasses must implement the create_table method.")

    def close_connection(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
