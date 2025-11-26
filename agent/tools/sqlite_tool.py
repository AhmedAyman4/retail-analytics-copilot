import sqlite3
import pandas as pd
from typing import List, Dict, Any, Tuple

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        self._create_friendly_views()

    def _create_friendly_views(self):
        """
        Creates views ONLY for tables with spaces/special characters.
        We do NOT create views for 'Orders' -> 'orders' because SQLite 
        treats them as name collisions.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. order_details (Needed to handle space in "Order Details")
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS order_details AS 
                SELECT OrderID, ProductID, UnitPrice, Quantity, Discount 
                FROM "Order Details"
            """)
            
            # Note: We skip creating views for Orders, Products, etc. 
            # because 'SELECT * FROM orders' works natively on the 'Orders' table.
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not create friendly views: {e}")

    def get_schema_info(self) -> str:
        """Introspects the DB and returns a schema string."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # We map the "Friendly Name" we want the LLM to use 
            # to the "Real Table Name" in the database.
            table_map = {
                'orders': 'Orders',
                'products': 'Products',
                'categories': 'Categories',
                'customers': 'Customers',
                'suppliers': 'Suppliers',
                'order_details': 'order_details', # This is our view
                'employees': 'Employees'
            }
            
            schema_str = "Database Schema (SQLite):\n"
            
            for friendly_name, real_name in table_map.items():
                # Check if the real table/view exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE (type='table' OR type='view') AND name='{real_name}';")
                if not cursor.fetchone():
                    continue
                    
                # We show the LLM the FRIENDLY name (lowercase) so it writes simple SQL
                schema_str += f"Table: {friendly_name}\n"
                
                # Get columns using the REAL name
                cursor.execute(f"PRAGMA table_info(\"{real_name}\");")
                columns = cursor.fetchall()
                for col in columns:
                    # col[1] is name, col[2] is type
                    schema_str += f"  - {col[1]} ({col[2]})\n"
                schema_str += "\n"
            
            conn.close()
            return schema_str
        except Exception as e:
            return f"Error retrieving schema: {str(e)}"

    def execute_query(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Executes a SQL query and returns (results, error_message).
        """
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            # Use pandas for safe execution and formatting
            df = pd.read_sql_query(query, conn)
            
            if df.empty:
                return [], None
                
            results = df.to_dict(orient='records')
            return results, None
        except Exception as e:
            return [], str(e)
        finally:
            if conn:
                conn.close()