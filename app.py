import streamlit as st
import dspy
import os
import sys
import time
import requests

# Ensure the root path is in pythonpath
sys.path.append(os.getcwd())

from agent.graph_hybrid import HybridAgent

# Page Config
st.set_page_config(page_title="Retail Analytics Copilot", layout="wide")

# --- Initialize Agent & Model (Cached) ---
@st.cache_resource
def load_agent_resources():
    """
    Connects to the local Ollama server and initializes the Agent Graph.
    """
    status = st.empty()
    status.info("Waiting for Ollama server to come online at localhost:11434...")
    
    # Wait for the server to be ready
    server_url = "http://localhost:11434"
    max_retries = 30 # Wait up to ~60 seconds
    server_ready = False
    
    for _ in range(max_retries):
        try:
            # Check if root endpoint returns "Ollama is running"
            response = requests.get(server_url)
            if response.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
        
    if not server_ready:
        status.error("Failed to connect to Ollama. Check logs.")
        return None

    status.info("Ollama online! Configuring DSPy...")
    
    try:
        # UPDATED: Configure DSPy for Ollama as requested
        lm = dspy.LM(
            model="ollama_chat/phi3.5:3.8b",  # Model tag
            api_base="http://localhost:11434", 
            api_key="", # Ollama doesn't require a key usually
        )
        
        dspy.configure(lm=lm)
        
        # Initialize the Graph
        agent_workflow = HybridAgent().build_graph()
        
        status.success("Agent Connected & Ready!")
        time.sleep(1)
        status.empty() 
        return agent_workflow
    except Exception as e:
        status.error(f"Failed to configure agent: {e}")
        return None

# Load the agent
agent_app = load_agent_resources()

# --- UI Layout ---
st.title("🛍️ Northwind Retail Analytics Copilot")
st.markdown("""
Ask questions about sales, products, marketing calendars, and KPIs.
*Backed by Ollama (phi3.5:3.8b) + SQLite*
""")

# Sidebar
with st.sidebar:
    st.header("Debug Info")
    st.info("Backend: Ollama (Port 11434)")
    if st.checkbox("Show Schema"):
        from agent.tools.sqlite_tool import SQLiteTool
        st.code(SQLiteTool().get_schema_info())

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("Details"):
                st.json(message["details"])

# Input
if prompt := st.chat_input("Ex: What was the AOV during Summer 1997?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if agent_app:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Thinking...")
            
            try:
                initial_state = {
                    "question": prompt,
                    "format_hint": "str",
                    "retries": 0,
                    "sql_error": None
                }
                
                output = agent_app.invoke(initial_state)
                
                final_ans = output.get("final_answer", "No answer.")
                explanation = output.get("explanation", "")
                
                full_response = f"**Answer:** {final_ans}\n\n_{explanation}_"
                placeholder.markdown(full_response)
                
                details = {
                    "sql": output.get("sql_query"),
                    "citations": output.get("citations"),
                    "sql_error": output.get("sql_error")
                }
                
                with st.expander("Details"):
                    st.json(details)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "details": details
                })
                
            except Exception as e:
                placeholder.error(f"Error: {str(e)}")