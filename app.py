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

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")

model_option = st.sidebar.radio(
    "Select Model Provider",
    ["Ollama (Local)", "Google Gemini"],
    index=0
)

api_key = None
if model_option == "Google Gemini":
    default_key = os.getenv("GOOGLE_API_KEY", "")
    api_key = st.sidebar.text_input("Enter Gemini API Key", value=default_key, type="password")
    if not api_key:
        st.sidebar.warning("⚠️ API Key required for Gemini")

# --- Initialize Agent & Model (Cached) ---
@st.cache_resource(show_spinner=False) 
def load_agent_resources(provider, gemini_key=None):
    status = st.empty()
    lm = None
    target_model_name = ""

    # --- OLLAMA SETUP ---
    if provider == "Ollama (Local)":
        target_model = "phi3.5:3.8b"
        target_model_name = target_model
        status.info(f"Waiting for Ollama server to come online at localhost:11434...")
        
        server_url = "http://localhost:11434"
        max_retries = 30
        server_ready = False
        
        for _ in range(max_retries):
            try:
                response = requests.get(server_url)
                if response.status_code == 200:
                    server_ready = True
                    break
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
            
        if not server_ready:
            status.error("Failed to connect to Ollama. Check logs.")
            return None, None, None

        status.info(f"Ollama is online. Checking for model '{target_model}'...")
        model_ready = False
        pull_retries = 150
        
        for i in range(pull_retries):
            try:
                tags_response = requests.get(f"{server_url}/api/tags")
                if tags_response.status_code == 200:
                    tags_data = tags_response.json()
                    models = [m.get('name') for m in tags_data.get('models', [])]
                    if any(target_model in m for m in models):
                        model_ready = True
                        break
            except Exception:
                pass
            time.sleep(2)

        if not model_ready:
            status.error("Model download timed out or failed.")
            return None, None, None

        try:
            lm = dspy.LM(
                model=f"ollama/{target_model}",
                api_base="http://localhost:11434", 
                api_key="",
            )
        except Exception as e:
            status.error(f"Failed to initialize Ollama LM: {e}")
            return None, None, None

    # --- GEMINI SETUP ---
    elif provider == "Google Gemini":
        target_model_name = "gemini-2.5-flash"
        if not gemini_key:
            return None, None, None 
        
        status.info("Connecting to Google Gemini...")
        try:
            lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_key)
        except Exception as e:
            status.error(f"Failed to initialize Gemini: {e}")
            return None, None, None

    status.info(f"Model ({target_model_name}) ready! Configuring Agent...")
    
    try:
        # REMOVED dspy.configure(lm=lm) to avoid thread errors
        # We will apply the LM using a context manager during execution
        
        # Instantiate the class
        agent_instance = HybridAgent()
        
        # Build the graph
        # Note: If build_graph makes LLM calls during init, we might need to wrap this too.
        # Assuming build_graph is structural only.
        agent_workflow = agent_instance.build_graph()
        
        status.success(f"Agent Connected & Ready! (Model: {target_model_name})")
        time.sleep(1)
        status.empty() 
        
        return agent_workflow, agent_instance, lm
        
    except Exception as e:
        status.error(f"Failed to configure agent: {e}")
        return None, None, None

# Load resources
agent_workflow, agent_instance, lm_instance = load_agent_resources(model_option, api_key)

# --- UI Layout ---
st.title("🛍️ Northwind Retail Analytics Copilot")

# Sidebar Debug
with st.sidebar:
    st.divider()
    st.header("Debug Info")
    
    if model_option == "Ollama (Local)":
        st.info("Backend: Ollama (Port 11434)")
        st.info("Model: phi3.5:3.8b")
    else:
        st.info("Backend: Google Vertex/AI Studio")
        st.info("Model: gemini-2.5-flash")
    
    if st.button("Test LLM Connection"):
        if lm_instance:
            with st.spinner("Testing simple prompt..."):
                try:
                    # Direct call to LM doesn't strictly need context, but good practice
                    res = lm_instance("Say hello!", max_tokens=10)
                    st.success(f"Success! Model replied: {res}")
                except Exception as e:
                    st.error(f"Connection Failed: {e}")
        else:
            st.warning("Model not loaded yet.")

    if st.checkbox("Show Schema"):
        try:
            from agent.tools.sqlite_tool import SQLiteTool
            st.code(SQLiteTool().get_schema_info())
        except ImportError:
            st.warning("SQLiteTool not found in path")

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

    if not agent_workflow:
        if model_option == "Google Gemini" and not api_key:
            st.error("Please enter a Gemini API Key in the sidebar.")
        else:
            st.error("Agent failed to load. Check settings.")
    else:
        with st.chat_message("assistant"):
            
            status_container = st.status("🚀 **Agent Working...**", expanded=True)
            
            def stream_callback(msg):
                status_container.write(msg)
            
            agent_instance.status_callback = stream_callback
            
            try:
                initial_state = {
                    "question": prompt,
                    "format_hint": "str",
                    "retries": 0,
                    "sql_error": None
                }
                
                # --- CRITICAL FIX: USE CONTEXT MANAGER ---
                # This applies the LM only for this block of code, avoiding global thread locking issues
                with dspy.context(lm=lm_instance):
                    output = agent_workflow.invoke(initial_state)
                
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
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": full_response,
                    "details": details
                })
                
            except Exception as e:
                status_container.update(label="❌ **Error**", state="error")
                st.error(f"An error occurred: {str(e)}")
            finally:
                agent_instance.status_callback = None