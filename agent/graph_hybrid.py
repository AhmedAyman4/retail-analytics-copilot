import dspy
from typing import TypedDict, List, Any, Dict, Literal, Callable, Optional
from langgraph.graph import StateGraph, END
import logging
import json
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from agent.dspy_signatures import RouterModule, PlannerModule, SQLModule, SynthesizerModule
from agent.tools.sqlite_tool import SQLiteTool
from agent.rag.retrieval import LocalRetriever

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

class HybridAgent:
    def __init__(self):
        self.sqlite = SQLiteTool()
        self.retriever = LocalRetriever()
        
        self.router = RouterModule()
        self.planner = PlannerModule()
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
        self.log(f"🚦 **Router:** Classifying intent for: '{state['question']}'...")
        result = self.router(question=state["question"])
        cls = result.classification.lower().strip()
        if cls not in ['sql', 'rag', 'hybrid']:
            cls = 'hybrid'
        self.log(f"✅ **Router:** Determined intent is **{cls.upper()}**")
        return {"classification": cls}

    def retriever_node(self, state: AgentState):
        self.log("📚 **Retriever:** Searching documentation...")
        chunks = self.retriever.search(state["question"], k=3)
        self.log(f"✅ **Retriever:** Found {len(chunks)} relevant documents.")
        
        if chunks:
            snippet = chunks[0]['text'][:150].replace('\n', ' ')
            self.log(f"📄 *Top Context:* \"{snippet}...\"")
            
        return {"rag_chunks": chunks}

    def planner_node(self, state: AgentState):
        self.log("🧠 **Planner:** Analyzing constraints...")
        context_text = "\n".join([c['text'] for c in state.get("rag_chunks", [])])
        result = self.planner(context=context_text, question=state["question"])
        return {"plan_requirements": result.sql_requirements}

    def sql_gen_node(self, state: AgentState):
        self.log("💾 **SQL Generator:** Writing query...")
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
        
        # --- HEURISTIC REPAIRS (Safety Net) ---
        # 1. Fix common typo BETWE0N -> BETWEEN
        clean_sql = clean_sql.replace("BETWE0N", "BETWEEN")
        
        # 2. Fix table names if model ignores friendly views
        clean_sql = clean_sql.replace("OrderDetails", "order_details")
        clean_sql = clean_sql.replace('"Order Details"', "order_details")
        clean_sql = clean_sql.replace("Orders", "orders")
        clean_sql = clean_sql.replace("Products", "products")
        clean_sql = clean_sql.replace("Categories", "categories")
        
        self.log(f"📝 **Generated SQL:**\n```sql\n{clean_sql}\n```")
        return {"sql_query": clean_sql}

    def sql_exec_node(self, state: AgentState):
        self.log("⚡ **Executor:** Running SQL...")
        results, error = self.sqlite.execute_query(state["sql_query"])
        if error:
            self.log(f"❌ **SQL Error:** {error}")
            return {"sql_error": error, "sql_result": [], "retries": state.get("retries", 0) + 1}
        else:
            self.log(f"✅ **Executor:** Query successful ({len(results)} rows).")
            return {"sql_result": results, "sql_error": None}

    def synthesizer_node(self, state: AgentState):
        self.log("✍️ **Synthesizer:** Formulating answer...")
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
            # Error recovery for JSON parsing
            err_str = str(e)
            self.log(f"⚠️ **JSON Parse Warning:** {err_str[:100]}... Attempting repair.")
            
            final_answer = "Error generating answer."
            explanation = "Failed to parse model output."
            
            json_match = re.search(r'LM Response: ({.*})', err_str, re.DOTALL)
            if json_match:
                raw_json = json_match.group(1)
                try:
                    raw_json = raw_json.replace("explanicn", "explanation")
                    data = json.loads(raw_json)
                    final_answer = data.get("final_answer", final_answer)
                    explanation = data.get("explanation", data.get("reasoning", explanation))
                    self.log("✅ **Repair Successful:** Extracted answer from malformed JSON.")
                except:
                    pass

        # Build citations
        citations = []
        for c in state.get("rag_chunks", []):
            citations.append(c['id'])
        if sql_q:
            for table in ["orders", "order_details", "products", "categories", "customers"]:
                if table.lower() in sql_q.lower():
                    citations.append(table)
        
        return {
            "final_answer": final_answer, 
            "explanation": explanation,
            "citations": list(set(citations))
        }

    def build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self.router_node)
        workflow.add_node("retriever", self.retriever_node)
        workflow.add_node("planner", self.planner_node)
        workflow.add_node("sql_gen", self.sql_gen_node)
        workflow.add_node("sql_exec", self.sql_exec_node)
        workflow.add_node("synthesizer", self.synthesizer_node)

        workflow.set_entry_point("router")

        workflow.add_conditional_edges("router", lambda x: x["classification"], 
                                     {"rag": "retriever", "sql": "sql_gen", "hybrid": "retriever"})
        
        workflow.add_conditional_edges("retriever", 
                                     lambda x: "planner" if x["classification"] == "hybrid" else "synthesizer",
                                     {"planner": "planner", "synthesizer": "synthesizer"})

        workflow.add_edge("planner", "sql_gen")
        workflow.add_edge("sql_gen", "sql_exec")

        def check_sql_status(state):
            if state.get("sql_error") and state.get("retries", 0) < 2:
                self.log("⚠️ **Repair:** SQL failed. Fixing...")
                return "retry"
            return "done"

        workflow.add_conditional_edges("sql_exec", check_sql_status, {"retry": "sql_gen", "done": "synthesizer"})
        workflow.add_edge("synthesizer", END)
        return workflow.compile()