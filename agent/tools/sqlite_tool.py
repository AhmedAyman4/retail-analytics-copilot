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
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Warning: Could not create friendly views: {e}")

    def get_schema_info(self) -> str:
        """
        Introspects the DB. 
        UPDATED: Now includes the newly added Views so the agent can use them
        instead of writing complex joins.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 1. Standard Tables we want exposed
            core_tables = [
                'Orders', 'Products', 'Categories', 'Customers', 
                'Suppliers', 'Employees', 'Shippers'
            ]
            
            # 2. Our "Friendly" View
            friendly_views = ['order_details']
            
            # 3. The new "Legacy/Access" Views (The ones you just added)
            # We query sqlite_master to find them dynamically
            cursor.execute("SELECT name FROM sqlite_master WHERE type='view' AND name NOT LIKE 'sqlite_%'")
            all_views = [row[0] for row in cursor.fetchall()]
            
            # Filter to ensure we include the ones from create.sql
            legacy_views = [
                v for v in all_views 
                if v not in friendly_views and v not in ['orders', 'products', 'customers'] # avoid simple duplicate views
            ]

            schema_str = "Database Schema (SQLite):\n"
            
            # -- A. Core Tables --
            schema_str += "--- CORE TABLES ---\n"
            for table in core_tables:
                cursor.execute(f"PRAGMA table_info(\"{table}\");")
                columns = cursor.fetchall()
                if columns:
                    schema_str += f"Table: {table}\n"
                    for col in columns:
                        schema_str += f"  - {col[1]} ({col[2]})\n"
                    schema_str += "\n"

            # -- B. Helper Views --
            schema_str += "--- HELPER VIEWS (Use these for simpler queries) ---\n"
            
            # Add order_details
            cursor.execute(f"PRAGMA table_info(\"order_details\");")
            cols = cursor.fetchall()
            if cols:
                schema_str += f"View: order_details (Simplifies 'Order Details')\n"
                for col in cols:
                    schema_str += f"  - {col[1]} ({col[2]})\n"
                schema_str += "\n"

            # Add the Legacy Views (High Value for the Agent)
            for view in legacy_views:
                cursor.execute(f"PRAGMA table_info(\"{view}\");")
                cols = cursor.fetchall()
                if cols:
                    schema_str += f"View: {view}\n"
                    for col in cols:
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