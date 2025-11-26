import dspy

# 1. Router
class RouterSignature(dspy.Signature):
    """
    Classify the user question into one of three categories:
    - 'sql': Requires database access (sales numbers, orders, customer data, revenue, counts).
    - 'rag': Requires looking up static text policies, calendars, definitions, or return windows.
    - 'hybrid': Requires both (e.g., "sales during Summer 1997" - needs calendar dates + DB).
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

# 4. Synthesizer (UPDATED)
class SynthesizerSignature(dspy.Signature):
    """
    Answer the user question based on the provided Context or SQL Results.
    
    CRITICAL RULES:
    1. If the answer is explicitly found in the 'context' (e.g., policies, return days, definitions), USE IT. 
    2. Do NOT say "requires SQL" if the answer is in the text context.
    3. Only rely on 'sql_result' if the question asks for calculated numbers (revenue, counts, averages).
    4. Ensure output matches 'format_hint' exactly (e.g., if int, return just the number).
    """
    question = dspy.InputField()
    context = dspy.InputField(desc="Retrieved text chunks (Trust this info!)")
    sql_query = dspy.InputField(desc="Executed SQL", optional=True)
    sql_result = dspy.InputField(desc="Result rows from DB", optional=True)
    format_hint = dspy.InputField(desc="Required output format (e.g., int, float, list)")
    
    explanation = dspy.OutputField(desc="Reasoning: 'Found return window in context...'")
    final_answer = dspy.OutputField(desc="The typed answer matching format_hint")

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
        # UPDATED: Changed from Predict to ChainOfThought
        # This forces the model to reason ("I see '14 days' in the text") before answering.
        self.prog = dspy.ChainOfThought(SynthesizerSignature)
    def forward(self, question, context, sql_query, sql_result, format_hint):
        return self.prog(question=question, context=context, sql_query=sql_query, sql_result=sql_result, format_hint=format_hint)