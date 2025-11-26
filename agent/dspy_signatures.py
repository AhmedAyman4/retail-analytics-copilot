import dspy

# 1. Router
class RouterSignature(dspy.Signature):
    """
    Classify the user question into one of three categories:
    - 'sql': Requires database access (sales numbers, orders, customer data, revenue).
    - 'rag': Requires looking up text policies, calendars, or definitions.
    - 'hybrid': Requires both (e.g., "sales during Summer 1997" - needs calendar dates + DB).
    """
    question = dspy.InputField()
    classification = dspy.OutputField(desc="One of: 'sql', 'rag', 'hybrid'")

# 2. Planner
class PlannerSignature(dspy.Signature):
    """
    Analyze the question and retrieved context to extract constraints for SQL generation.
    - Map text terms (e.g., "Summer 1997") to specific DATE RANGES (YYYY-MM-DD).
    - Map text categories (e.g., "Beverages") to exact DB category names.
    """
    context = dspy.InputField(desc="Retrieved chunks from docs")
    question = dspy.InputField()
    analysis = dspy.OutputField(desc="Reasoning about dates, IDs, and formulas")
    sql_requirements = dspy.OutputField(desc="Specific WHERE clause constraints (e.g. OrderDate BETWEEN '1997-06-01' AND ...)")

# 3. Text to SQL (UPDATED)
class TextToSQLSignature(dspy.Signature):
    """
    Generate executable SQLite query for the Northwind database.
    
    CRITICAL RULES:
    1. Use these view names: 'orders', 'order_details', 'products', 'categories'.
    2. Date Format: Use string comparison. Example: OrderDate >= '1997-01-01'
    3. Do NOT use julianday() or complex functions.
    4. For Revenue: SUM(UnitPrice * Quantity * (1 - Discount))
    5. JOIN correctly: products ON order_details.ProductID = products.ProductID
    """
    schema = dspy.InputField()
    requirements = dspy.InputField(desc="Constraints from planner")
    question = dspy.InputField()
    previous_error = dspy.InputField(desc="Error from previous attempt, if any", optional=True)
    sql_query = dspy.OutputField(desc="The SQL query string only (start with SELECT)")

# 4. Synthesizer (Robust)
class SynthesizerSignature(dspy.Signature):
    """
    Answer the user question based on the provided Context or SQL Results.
    
    CRITICAL RULES:
    1. If the answer is in 'context', USE IT. Do not ask for SQL.
    2. Output strict JSON format with keys: "explanation", "final_answer".
    3. CHECK SPELLING: Key must be "explanation", NOT "explanicn".
    """
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved text chunks (Trust this info!)")
    sql_query = dspy.InputField(desc="Executed SQL", optional=True)
    sql_result = dspy.InputField(desc="Result rows from DB", optional=True)
    format_hint = dspy.InputField(desc="Required output format (e.g., int, float, list)")
    
    explanation = dspy.OutputField(desc="Brief reasoning (Key: 'explanation')")
    final_answer = dspy.OutputField(desc="The final answer (Key: 'final_answer')")

# Define Modules
class RouterModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.Predict(RouterSignature)
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