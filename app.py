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
    status = st.empty()
    status.info("Waiting for Ollama server to come online at localhost:11434...")
    
    server_url = "http://localhost:11434"
    max_retries = 30
    server_ready = False
    
    # 1. Wait for Ollama Server
    for _ in range(max_retries):
        try:
            response = requests.get(server_url)
            if response.status_code == 200:
                server_ready = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(2)
        
    if not server_ready:
        status.error("Failed to connect to Ollama. Check logs.")
        return None, None, None

    # 2. Wait for Model Download (since we backgrounded the pull)
    status.info("Ollama is online. Checking for model 'phi3.5:3.8b'...")
    model_ready = False
    pull_retries = 150 # Wait up to 5 minutes for download
    
    for i in range(pull_retries):
        try:
            # Check installed tags
            tags_response = requests.get(f"{server_url}/api/tags")
            if tags_response.status_code == 200:
                tags_data = tags_response.json()
                models = [m.get('name') for m in tags_data.get('models', [])]
                # Check for exact or partial match
                if any("phi3.5:3.8b" in m for m in models):
                    model_ready = True
                    break
        except Exception:
            pass
        
        status.info(f"Downloading model 'phi3.5:3.8b'... (Time elapsed: {i*2}s)")
        time.sleep(2)

    if not model_ready:
        status.error("Model download timed out or failed. Please restart the Space.")
        return None, None, None

    status.info("Model ready! Configuring Agent...")
    
    try:
        lm = dspy.LM(
            model="ollama/phi3.5:3.8b",
            api_base="http://localhost:11434", 
            api_key="",
        )
        
        dspy.configure(lm=lm)
        
        # Instantiate the class
        agent_instance = HybridAgent()
        # Build the graph
        agent_workflow = agent_instance.build_graph()
        
        status.success("Agent Connected & Ready!")
        time.sleep(1)
        status.empty() 
        
        # Return Workflow (runnable), Instance (for callbacks), and LM (for testing)
        return agent_workflow, agent_instance, lm
        
    except Exception as e:
        status.error(f"Failed to configure agent: {e}")
        return None, None, None

# Load the agent
agent_workflow, agent_instance, lm_instance = load_agent_resources()

# --- UI Layout ---
st.title("🛍️ Northwind Retail Analytics Copilot")

# Sidebar
with st.sidebar:
    st.header("Debug Info")
    st.info("Backend: Ollama (Port 11434)")
    
    if st.button("Test LLM Connection"):
        with st.spinner("Testing simple prompt..."):
            try:
                # Simple generation test
                res = lm_instance("Say hello!", max_tokens=10)
                st.success(f"Success! Model replied: {res}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

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
            with st.expander("Show Logic & Data"):
                st.json(message["details"])

# Input
if prompt := st.chat_input("Ex: What was the AOV during Summer 1997?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if agent_workflow and agent_instance:
        with st.chat_message("assistant"):
            
            # Create a Status Container to stream steps
            status_container = st.status("🚀 **Agent Working...**", expanded=True)
            
            # Define the callback function that writes to the status container
            def stream_callback(msg):
                status_container.write(msg)
            
            # Attach callback to the live instance
            agent_instance.status_callback = stream_callback
            
            try:
                initial_state = {
                    "question": prompt,
                    "format_hint": "str",
                    "retries": 0,
                    "sql_error": None
                }
                
                # Run the Agent
                output = agent_workflow.invoke(initial_state)
                
                # Update Status to complete
                status_container.update(label="✅ **Analysis Complete!**", state="complete", expanded=False)
                
                final_ans = output.get("final_answer", "No answer.")
                explanation = output.get("explanation", "")
                
                full_response = f"**Answer:** {final_ans}\n\n_{explanation}_"
                st.markdown(full_response)
                
                details = {
                    "sql": output.get("sql_query"),
                    "citations": output.get("citations"),
                    "sql_error": output.get("sql_error"),
                    "classification": output.get("classification")
                }
                
                # Save context
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "details": details
                })
                
            except Exception as e:
                status_container.update(label="❌ **Error**", state="error")
                st.error(f"An error occurred: {str(e)}")
            finally:
                # Cleanup callback
                agent_instance.status_callback = None