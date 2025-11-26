import dspy

# 1. Router
class RouterSignature(dspy.Signature):
    """
    Classify the user question into one of three categories:
    - 'sql': Requires database access (sales numbers, orders, customer data).
    - 'rag': Requires looking up text policies, calendars, or definitions.
    - 'hybrid': Requires both (e.g., "sales during Summer 1997").
    """
    question = dspy.InputField()
    classification = dspy.OutputField(desc="One of: 'sql', 'rag', 'hybrid'")

# 2. Planner
class PlannerSignature(dspy.Signature):
    """
    Analyze the question and retrieved context to extract constraints for SQL generation.
    Identify date ranges, product categories, or specific KPI formulas needed.
    """
    context = dspy.InputField(desc="Retrieved chunks from docs")
    question = dspy.InputField()
    analysis = dspy.OutputField(desc="Reasoning about dates, IDs, and formulas")
    sql_requirements = dspy.OutputField(desc="Specific filtering logic needed for SQL")

# 3. Text to SQL
class TextToSQLSignature(dspy.Signature):
    """
    Generate executable SQLite query for the Northwind database.
    Use the provided schema.
    Rules:
    - Use 'Order Details' table (quote it).
    - For Revenue: SUM(UnitPrice * Quantity * (1 - Discount)).
    - For Margin: SUM((UnitPrice * 0.7) * Quantity * (1 - Discount)) unless CostOfGoods exists.
    - Dates are in YYYY-MM-DD format.
    """
    schema = dspy.InputField()
    requirements = dspy.InputField(desc="Constraints from planner")
    question = dspy.InputField()
    previous_error = dspy.InputField(desc="Error from previous attempt, if any", optional=True)
    sql_query = dspy.OutputField(desc="The SQL query string only")

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
        # Using Predict instead of ChainOfThought to reduce JSON nesting issues
        # The prompt explicitly asks for reasoning in the 'explanation' field.
        self.prog = dspy.Predict(SynthesizerSignature)
    def forward(self, question, context, sql_query, sql_result, format_hint):
        return self.prog(question=question, context=context, sql_query=sql_query, sql_result=sql_result, format_hint=format_hint)