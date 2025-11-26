import sqlite3
import pandas as pd
from typing import List, Dict, Any, Tuple

class SQLiteTool:
    def __init__(self, db_path: str = "data/northwind.sqlite"):
        self.db_path = db_path
        self._create_friendly_views()

    def _create_friendly_views(self):
        """
        Creates lowercase, space-free views to make SQL generation easier 
        for small models (e.g., 'Order Details' -> 'order_details').
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. order_details
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS order_details AS 
                SELECT OrderID, ProductID, UnitPrice, Quantity, Discount 
                FROM "Order Details"
            """)
            
            # 2. orders (lowercase)
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS orders AS 
                SELECT * FROM Orders
            """)
            
            # 3. products (lowercase)
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS products AS 
                SELECT * FROM Products
            """)
            
            # 4. categories (lowercase)
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS categories AS 
                SELECT * FROM Categories
            """)

            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not create friendly views: {e}")

    def get_schema_info(self) -> str:
        """Introspects the DB and returns a schema string."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Only show our friendly views to the model to keep it focused
            views = ['orders', 'order_details', 'products', 'categories', 'customers', 'suppliers']
            
            schema_str = "Database Schema (SQLite):\n"
            
            for table_name in views:
                # Check if table/view exists
                cursor.execute(f"SELECT name FROM sqlite_master WHERE (type='table' OR type='view') AND name='{table_name}';")
                if not cursor.fetchone():
                    continue
                    
                schema_str += f"Table: {table_name}\n"
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