import dspy
from typing import TypedDict, List, Any, Dict, Literal, Callable, Optional
from langgraph.graph import StateGraph, END
import logging
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agent.dspy_signatures import RouterModule, SQLModule, SynthesizerModule
from agent.tools.sqlite_tool import SQLiteTool
from agent.rag.retrieval import LocalRetriever

class AgentState(TypedDict):
    question: str
    format_hint: str
    classification: str
    rag_chunks: List[Dict]
    sql_query: str
    sql_result: List[Dict]
    sql_error: str
    final_answer: Any
    explanation: str
    citations: List[str]
    retries: int

class HybridAgent:
    def __init__(self):
        self.sqlite = SQLiteTool()
        self.retriever = LocalRetriever()
        
        self.router = RouterModule()
        self.sql_gen = SQLModule()
        self.synth = SynthesizerModule()
        
        self.status_callback: Optional[Callable[[str], None]] = None

    def log(self, message: str):
        logger.info(message)
        if self.status_callback:
            try:
                self.status_callback(message)
            except Exception:
                pass

    def router_node(self, state: AgentState):
        self.log(f"🚦 **Router:** Classifying: '{state['question']}'")
        
        # 1. Run LLM Router (Now with ChainOfThought)
        result = self.router(question=state["question"])
        
        # Log reasoning if available
        if hasattr(result, 'reasoning'):
             self.log(f"🤔 **Reasoning:** {result.reasoning}")
             
        cls = result.classification.lower().strip()
        
        # 2. HEURISTIC OVERRIDE (Safety Net)
        # Small models often miss that "defined in calendar" means they need to look up the calendar.
        keywords = ["defined in", "calendar", "policy", "return window", "terms", "marketing"]
        if any(k in state['question'].lower() for k in keywords):
            if cls == 'sql':
                self.log(f"⚠️ **Override:** Keyword detected. Switching SQL -> HYBRID.")
                cls = 'hybrid'
        
        if cls not in ['sql', 'rag', 'hybrid']:
            cls = 'hybrid'
            
        self.log(f"✅ **Result:** {cls.upper()}")
        return {"classification": cls}

    def retriever_node(self, state: AgentState):
        self.log("📚 **Retriever:** Searching docs...")
        chunks = self.retriever.search(state["question"], k=3)
        return {"rag_chunks": chunks}

    def sql_gen_node(self, state: AgentState):
        self.log("💾 **SQL Generator:** Writing query...")
        schema = self.sqlite.get_schema_info()
        
        # Pass context directly
        context_text = "\n".join([c['text'] for c in state.get("rag_chunks", [])])
        
        pred = self.sql_gen(
            schema=schema, 
            requirements=f"Context: {context_text}", 
            question=state["question"],
            previous_error=state.get("sql_error", "")
        )
        
        # --- AGGRESSIVE CLEANING ---
        raw_sql = pred.sql_query
        # 1. Remove markdown
        clean_sql = raw_sql.replace("```sql", "").replace("```", "").strip()
        
        # 2. Remove comments (Lines starting with -- or //)
        clean_sql = re.sub(r'--.*', '', clean_sql)
        clean_sql = re.sub(r'//.*', '', clean_sql)
        # 3. Remove block comments (/* ... */)
        clean_sql = re.sub(r'/\*.*?\*/', '', clean_sql, flags=re.DOTALL)
        
        # --- HEURISTIC REPAIRS ---
        clean_sql = clean_sql.replace("BETWE0N", "BETWEEN")
        
        # Fix Table Names (Map model's guess to our views/tables)
        clean_sql = clean_sql.replace("OrderDetails", "order_details")
        clean_sql = clean_sql.replace('"Order Details"', "order_details")
        
        # CRITICAL: Fix the "orders.ProductID" hallucination
        if "orders.ProductID" in clean_sql:
             self.log("⚠️ **Patching:** Detected invalid Join (orders -> products). Fixing...")
             clean_sql = clean_sql.replace("orders.ProductID", "order_details.ProductID")
        
        # Fix Date Column
        clean_sql = clean_sql.replace("ShipDate", "OrderDate")
        clean_sql = clean_sql.replace("o.ShipDate", "o.OrderDate")
        
        self.log(f"📝 **SQL:** `{clean_sql}`")
        return {"sql_query": clean_sql}

    def sql_exec_node(self, state: AgentState):
        self.log("⚡ **Executor:** Running SQL...")
        results, error = self.sqlite.execute_query(state["sql_query"])
        if error:
            self.log(f"❌ Error: {error}")
            return {"sql_error": error, "sql_result": [], "retries": state.get("retries", 0) + 1}
        else:
            self.log(f"✅ Success: {len(results)} rows.")
            return {"sql_result": results, "sql_error": None}

    def synthesizer_node(self, state: AgentState):
        self.log("✍️ **Synthesizer:** Answering...")
        context_text = "\n".join([c['text'] for c in state.get("rag_chunks", [])])
        sql_q = state.get("sql_query", "")
        sql_res = str(state.get("sql_result", []))
        
        try:
            pred = self.synth(
                question=state["question"],
                context=context_text,
                sql_query=sql_q,
                sql_result=sql_res,
                format_hint=state["format_hint"]
            )
            final_answer = pred.final_answer
            explanation = pred.explanation
        except Exception as e:
            # Quick repair logic
            final_answer = "Error"
            explanation = "Parse Error"
            try:
                raw_json = str(e).split("LM Response:")[-1]
                raw_json = raw_json.replace("explanicn", "explanation")
                data = json.loads(raw_json)
                final_answer = data.get("final_answer")
                explanation = data.get("explanation")
            except:
                pass

        citations = [c['id'] for c in state.get("rag_chunks", [])]
        
        return {
            "final_answer": final_answer, 
            "explanation": explanation,
            "citations": citations
        }

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("sql_gen", self.sql_gen_node)
        workflow.add_node("sql_exec", self.sql_exec_node)
        workflow.add_node("synthesizer", self.synthesizer_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges("router", lambda x: x["classification"], 
                                     {"rag": "retriever", "sql": "sql_gen", "hybrid": "retriever"})
        
        workflow.add_conditional_edges("retriever", 
                                     lambda x: "sql_gen" if x["classification"] == "hybrid" else "synthesizer",
                                     {"sql_gen": "sql_gen", "synthesizer": "synthesizer"})

        workflow.add_edge("sql_gen", "sql_exec")

        def check_sql_status(state):
            if state.get("sql_error") and state.get("retries", 0) < 2:
                return "retry"
            return "done"

        workflow.add_conditional_edges("sql_exec", check_sql_status, {"retry": "sql_gen", "done": "synthesizer"})
        workflow.add_edge("synthesizer", END)
        return workflow.compile()