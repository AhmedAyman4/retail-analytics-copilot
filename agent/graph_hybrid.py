import dspy
from typing import TypedDict, List, Any, Dict, Literal
from langgraph.graph import StateGraph, END
import logging

# Setup basic logging to console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agent.dspy_signatures import RouterModule, PlannerModule, SQLModule, SynthesizerModule
from agent.tools.sqlite_tool import SQLiteTool
from agent.rag.retrieval import LocalRetriever

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    format_hint: str
    
    classification: str
    rag_chunks: List[Dict]
    plan_requirements: str
    
    sql_query: str
    sql_result: List[Dict]
    sql_error: str
    
    final_answer: Any
    explanation: str
    citations: List[str]
    
    retries: int

# --- Node Logic ---

class HybridAgent:
    def __init__(self):
        self.sqlite = SQLiteTool()
        self.retriever = LocalRetriever()
        
        self.router = RouterModule()
        self.planner = PlannerModule()
        self.sql_gen = SQLModule()
        self.synth = SynthesizerModule()

    def router_node(self, state: AgentState):
        logger.info(f"--- [Node: Router] Classifying: {state['question']} ---")
        result = self.router(question=state["question"])
        cls = result.classification.lower().strip()
        if cls not in ['sql', 'rag', 'hybrid']:
            cls = 'hybrid'
        logger.info(f"--- Classification: {cls} ---")
        return {"classification": cls}

    def retriever_node(self, state: AgentState):
        logger.info("--- [Node: Retriever] Searching Docs ---")
        chunks = self.retriever.search(state["question"], k=3)
        return {"rag_chunks": chunks}

    def planner_node(self, state: AgentState):
        logger.info("--- [Node: Planner] Analyzing Constraints ---")
        context_text = "\n".join([c['text'] for c in state.get("rag_chunks", [])])
        result = self.planner(context=context_text, question=state["question"])
        return {"plan_requirements": result.sql_requirements}

    def sql_gen_node(self, state: AgentState):
        logger.info("--- [Node: SQL Gen] Generating Query ---")
        schema = self.sqlite.get_schema_info()
        reqs = state.get("plan_requirements", "")
        err = state.get("sql_error", "")
        
        pred = self.sql_gen(
            schema=schema, 
            requirements=reqs, 
            question=state["question"],
            previous_error=err
        )
        
        clean_sql = pred.sql_query.replace("```sql", "").replace("```", "").strip()
        logger.info(f"Generated SQL: {clean_sql}")
        return {"sql_query": clean_sql}

    def sql_exec_node(self, state: AgentState):
        logger.info("--- [Node: SQL Exec] Running Query ---")
        results, error = self.sqlite.execute_query(state["sql_query"])
        if error:
            logger.error(f"SQL Failed: {error}")
            return {"sql_error": error, "sql_result": [], "retries": state.get("retries", 0) + 1}
        else:
            logger.info(f"SQL Success: {len(results)} rows found")
            return {"sql_result": results, "sql_error": None}

    def synthesizer_node(self, state: AgentState):
        logger.info("--- [Node: Synthesizer] Formatting Answer ---")
        context_text = "\n".join([c['text'] for c in state.get("rag_chunks", [])])
        sql_q = state.get("sql_query", "")
        sql_res = str(state.get("sql_result", []))
        
        pred = self.synth(
            question=state["question"],
            context=context_text,
            sql_query=sql_q,
            sql_result=sql_res,
            format_hint=state["format_hint"]
        )
        
        citations = []
        for c in state.get("rag_chunks", []):
            citations.append(c['id'])
        if sql_q:
            for table in ["Orders", "Order Details", "Products", "Customers"]:
                if table.lower() in sql_q.lower() or f'"{table}"' in sql_q:
                    citations.append(table)
        
        return {
            "final_answer": pred.final_answer, 
            "explanation": pred.explanation,
            "citations": list(set(citations))
        }

    # --- Graph Construction ---
    def build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("sql_gen", self.sql_gen_node)
        workflow.add_node("sql_exec", self.sql_exec_node)
        workflow.add_node("synthesizer", self.synthesizer_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges(
            "router",
            lambda x: x["classification"],
            {
                "rag": "retriever",
                "sql": "sql_gen",
                "hybrid": "retriever"
            }
        )
        
        workflow.add_conditional_edges(
            "retriever",
            lambda x: "planner" if x["classification"] == "hybrid" else "synthesizer",
            {
                "planner": "planner",
                "synthesizer": "synthesizer"
            }
        )

        workflow.add_edge("planner", "sql_gen")
        workflow.add_edge("sql_gen", "sql_exec")

        def check_sql_status(state):
            if state.get("sql_error") and state.get("retries", 0) < 2:
                logger.warning("Triggering Repair Loop...")
                return "retry"
            return "done"

        workflow.add_conditional_edges(
            "sql_exec",
            check_sql_status,
            {
                "retry": "sql_gen",
                "done": "synthesizer"
            }
        )

        workflow.add_edge("synthesizer", END)

        return workflow.compile()