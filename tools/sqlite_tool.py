import sqlite3
import pandas as pd
from typing import List, Dict, Any, Tuple

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path

    def get_schema_info(self) -> str:
        """Introspects the DB and returns a string describing tables and columns."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            
            schema_str = ""
            for table in tables:
                table_name = table[0]
                schema_str += f"Table: {table_name}\n"
                
                # Get columns for each table
                cursor.execute(f"PRAGMA table_info(\"{table_name}\");")
                columns = cursor.fetchall()
                for col in columns:
                    # cid, name, type, notnull, dflt_value, pk
                    schema_str += f"  - {col[1]} ({col[2]})\n"
                schema_str += "\n"
            
            conn.close()
            return schema_str
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Executes a SQL query and returns (results, error_message).
        If success, error_message is None.
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Use pandas for easy result formatting, though raw cursor works too
            df = pd.read_sql_query(query, conn)
            results = df.to_dict(orient='records')
            return results, None
        except Exception as e:
            return [], str(e)
        finally:
            if conn:
                conn.close()