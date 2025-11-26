import streamlit as st
import dspy
import os
import json
from agent.graph_hybrid import HybridAgent

# Page Config
st.set_page_config(page_title="Retail Analytics Copilot", layout="wide")

# --- Initialize Agent & Model (Cached) ---
@st.cache_resource
def load_agent_resources():
    """
    Loads the Hugging Face model and initializes the Agent Graph.
    This is cached to prevent reloading the heavy model on every interaction.
    """
    status = st.empty()
    status.info("Loading Microsoft Phi-3.5 model... (This may take a minute)")
    
    # Load Model (CPU friendly-ish, but requires ~8GB RAM)
    try:
        lm = dspy.HFModel(model='microsoft/Phi-3.5-mini-instruct')
        dspy.settings.configure(lm=lm)
        
        # Initialize the Graph
        agent_workflow = HybridAgent().build_graph()
        
        status.success("Model and Agent loaded successfully!")
        status.empty() # Clear the status message
        return agent_workflow
    except Exception as e:
        status.error(f"Failed to load model: {e}")
        return None

# Load the agent
agent_app = load_agent_resources()

# --- UI Layout ---
st.title("🛍️ Northwind Retail Analytics Copilot")
st.markdown("""
Ask questions about sales, products, marketing calendars, and KPIs.
*Backed by local RAG + SQLite + Phi-3.5*
""")

# Sidebar for debug/info
with st.sidebar:
    st.header("Debug Info")
    st.info("Model: microsoft/Phi-3.5-mini-instruct")
    if st.checkbox("Show Schema"):
        from agent.tools.sqlite_tool import SQLiteTool
        st.code(SQLiteTool().get_schema_info())

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "details" in message:
            with st.expander("Details (SQL & Citations)"):
                st.json(message["details"])

# Chat Input
if prompt := st.chat_input("Ex: What was the AOV during Summer 1997?"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    if agent_app:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # Initial State for the Graph
                initial_state = {
                    "question": prompt,
                    "format_hint": "str", # Default to string for chat interface
                    "retries": 0,
                    "sql_error": None
                }
                
                # Run the Agent
                output_state = agent_app.invoke(initial_state)
                
                # Parse Output
                final_ans = output_state.get("final_answer", "No answer generated.")
                explanation = output_state.get("explanation", "")
                
                # Construct display text
                full_response = f"**Answer:** {final_ans}\n\n_{explanation}_"
                message_placeholder.markdown(full_response)
                
                # Prepare details for the expander
                details = {
                    "sql_query": output_state.get("sql_query"),
                    "citations": output_state.get("citations"),
                    "classification": output_state.get("classification"),
                    "sql_error": output_state.get("sql_error")
                }
                
                with st.expander("Details (SQL & Citations)"):
                    st.json(details)
                
                # Save assistant response to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "details": details
                })
                
            except Exception as e:
                message_placeholder.error(f"An error occurred: {str(e)}")