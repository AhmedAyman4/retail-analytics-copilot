import dspy

# 1. Router Signature (Updated for CoT)
class RouterSignature(dspy.Signature):
    """
    Classify the user question into one of three categories:
    - 'sql': Pure database queries (e.g. "count orders", "total revenue"). Use ONLY if dates/ids are explicit (e.g. "1997-01-01").
    - 'rag': Pure text lookups (e.g. "what is the return policy", "meaning of AOV").
    - 'hybrid': Queries requiring both Data and Docs.
    """
    question = dspy.InputField()
    classification = dspy.OutputField(desc="One of: 'sql', 'rag', 'hybrid'")

# 2. Planner Signature (Restored)
class PlannerSignature(dspy.Signature):
    """
    Analyze the question and retrieved context to extract constraints for SQL generation.
    Identify date ranges, product categories, or specific KPI formulas needed.
    """
    context = dspy.InputField(desc="Retrieved chunks from docs")
    question = dspy.InputField()
    analysis = dspy.OutputField(desc="Reasoning about dates, IDs, and formulas")
    sql_requirements = dspy.OutputField(desc="Specific filtering logic needed for SQL")

# 3. SQL Signature
class TextToSQLSignature(dspy.Signature):
    """
    Generate executable SQLite query for the Northwind database.
    
    SCHEMA CHEAT SHEET (FOLLOW STRICTLY):
    1. JOINS:
       - orders -> order_details: ON orders.OrderID = order_details.OrderID
       - order_details -> products: ON order_details.ProductID = products.ProductID
       - products -> categories: ON products.CategoryID = categories.CategoryID
       
    2. COLUMNS:
       - Revenue Logic: SUM(order_details.UnitPrice * order_details.Quantity * (1 - order_details.Discount))
       - Date Column: Use 'orders.OrderDate' (Format: 'YYYY-MM-DD'). DO NOT USE ShipDate.
       - Product Names: Use 'products.ProductName'.
       - Category Names: Use 'categories.CategoryName'.
       
    3. RULES:
       - return ONLY the SQL string.
       - NO comments (-- or //).
       - NO explanations.
    """
    schema = dspy.InputField()
    requirements = dspy.InputField(desc="Context info or requirements")
    question = dspy.InputField()
    previous_error = dspy.InputField(desc="Error from previous attempt", optional=True)
    sql_query = dspy.OutputField(desc="The SQL query string only")

# 4. Synthesizer Signature
class SynthesizerSignature(dspy.Signature):
    """
    Answer based on Context or SQL Results.
    Output JSON with keys: "explanation", "final_answer".
    """
    question = dspy.InputField()
    context = dspy.InputField()
    sql_query = dspy.InputField(optional=True)
    sql_result = dspy.InputField(optional=True)
    format_hint = dspy.InputField()
    
    explanation = dspy.OutputField(desc="Reasoning")
    final_answer = dspy.OutputField(desc="Final answer")

# --- MODULES ---

class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        # UPDATED: Use ChainOfThought!
        self.prog = dspy.ChainOfThought(RouterSignature)
    def forward(self, question):
        return self.prog(question=question)

class PlannerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(PlannerSignature)
    def forward(self, context, question):
        return self.prog(context=context, question=question)

class SQLModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(TextToSQLSignature)
    def forward(self, schema, requirements, question, previous_error=""):
        return self.prog(schema=schema, requirements=requirements, question=question, previous_error=previous_error)

class SynthesizerModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(SynthesizerSignature)
    def forward(self, question, context, sql_query, sql_result, format_hint):
        return self.prog(question=question, context=context, sql_query=sql_query, sql_result=sql_result, format_hint=format_hint)