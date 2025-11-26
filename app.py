import streamlit as st
import dspy
import os
import sys
import time
import requests
import json
import ast

# Ensure the root path is in pythonpath
sys.path.append(os.getcwd())

from agent.graph_hybrid import HybridAgent

# Page Config
st.set_page_config(page_title="Retail Analytics Copilot", layout="wide")

# --- Sidebar Configuration ---
st.sidebar.title("⚙️ Configuration")

# 1. App Mode Selection
app_mode = st.sidebar.selectbox("App Mode", ["Interactive Chat", "Batch Evaluation"])

# 2. Model Provider Selection
st.sidebar.divider()
model_option = st.sidebar.radio(
    "Select Model Provider",
    ["Ollama (Local)", "Google Gemini"],
    index=0
)

gemini_key = None
if model_option == "Google Gemini":
    env_key = os.getenv("GOOGLE_API_KEY")
    secrets_key = None
    try:
        if "GOOGLE_API_KEY" in st.secrets:
            secrets_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        pass 

    gemini_key = env_key or secrets_key
    
    if gemini_key:
        st.sidebar.success("✅ API Key active")
    else:
        st.sidebar.warning("Secret 'GOOGLE_API_KEY' not found.")
        gemini_key = st.sidebar.text_input("Enter Gemini API Key manually", type="password")

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
        status.info(f"Waiting for Ollama server...")
        
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
            status.error("Failed to connect to Ollama.")
            return None, None, None

        status.info(f"Ollama online. Checking model '{target_model}'...")
        # Check model existence (Simplified for speed)
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
        try:
            lm = dspy.LM("gemini/gemini-2.5-flash", api_key=gemini_key)
        except Exception as e:
            status.error(f"Failed to initialize Gemini: {e}")
            return None, None, None

    status.info(f"Model ready! Configuring Agent...")
    
    try:
        agent_instance = HybridAgent()
        agent_workflow = agent_instance.build_graph()
        status.success(f"Ready! ({target_model_name})")
        time.sleep(1)
        status.empty() 
        return agent_workflow, agent_instance, lm
        
    except Exception as e:
        status.error(f"Failed to configure agent: {e}")
        return None, None, None

# Load resources
agent_workflow, agent_instance, lm_instance = load_agent_resources(model_option, gemini_key)

# --- Main Title ---
st.title("🛍️ Northwind Retail Analytics Copilot")

# ==========================================
# MODE 1: INTERACTIVE CHAT
# ==========================================
if app_mode == "Interactive Chat":
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "details" in message:
                with st.expander("Show Logic & Data"):
                    st.json(message["details"])

    if prompt := st.chat_input("Ex: What was the AOV during Summer 1997?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        if not agent_workflow:
            st.error("Agent failed to load.")
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

# ==========================================
# MODE 2: BATCH EVALUATION
# ==========================================
elif app_mode == "Batch Evaluation":
    st.header("📂 Batch File Evaluation")
    st.markdown("Upload a `.jsonl` file to run the agent on multiple questions.")

    uploaded_file = st.file_uploader("Upload Questions", type=["jsonl", "json"])

    if uploaded_file and agent_workflow:
        # Load and Preview Data
        try:
            lines = uploaded_file.getvalue().decode("utf-8").strip().split('\n')
            questions = [json.loads(line) for line in lines if line.strip()]
            st.success(f"Loaded {len(questions)} questions.")
            
            with st.expander("Preview Input Data"):
                st.json(questions[:2])
                
        except Exception as e:
            st.error(f"Error parsing file: {e}")
            questions = []

        if questions:
            if st.button(f"Run Agent on {len(questions)} Questions"):
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # --- PROCESSING LOOP ---
                for idx, item in enumerate(questions):
                    status_text.text(f"Processing {idx+1}/{len(questions)}: {item.get('id')}...")
                    
                    try:
                        # Prepare State
                        initial_state = {
                            "question": item['question'],
                            "format_hint": item.get('format_hint', 'str'),
                            "retries": 0,
                            "sql_error": None
                        }
                        
                        # Run Agent (Silent Mode - no callbacks to speed up)
                        agent_instance.status_callback = None 
                        
                        output = None
                        
                        # --- RETRY LOGIC FOR RATE LIMITS ---
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                with dspy.context(lm=lm_instance):
                                    output = agent_workflow.invoke(initial_state)
                                break # Success, exit retry loop
                            except Exception as e:
                                err_str = str(e).lower()
                                # Check for Rate Limit (429) errors from Gemini
                                if "429" in err_str or "quota" in err_str:
                                    if attempt < max_retries - 1:
                                        wait_time = 60 # Gemini usually needs ~60s
                                        status_text.warning(f"Rate limit hit on {item.get('id')}. Sleeping {wait_time}s... (Attempt {attempt+1})")
                                        time.sleep(wait_time)
                                        status_text.text(f"Retrying {item.get('id')}...")
                                    else:
                                        raise e # Give up after retries
                                else:
                                    raise e # Not a rate limit error, raise immediately

                        if output:
                            # Parse Final Answer (Handle types)
                            final_ans = output.get("final_answer")
                            
                            # --- FIX: Check the correct variable 'final_ans' ---
                            if isinstance(final_ans, str) and item.get('format_hint') != 'str':
                                 try:
                                     # Clean up markdown code blocks if present
                                     clean_ans = final_ans.replace("```json", "").replace("```", "").strip()
                                     # Heuristic: try ast.literal_eval if it looks like a structure
                                     if "{" in clean_ans or "[" in clean_ans:
                                         final_ans = ast.literal_eval(clean_ans)
                                 except:
                                     pass

                            # Calculate Confidence Heuristic
                            confidence = 0.7
                            if output.get("sql_error"):
                                confidence = 0.1
                            elif output.get("classification") == "sql" and output.get("sql_result"):
                                confidence = 0.9

                            # Build Record
                            record = {
                                "id": item['id'],
                                "final_answer": final_ans,
                                "sql": output.get("sql_query", ""),
                                "confidence": confidence,
                                "explanation": output.get("explanation", ""),
                                "citations": output.get("citations", [])
                            }
                            results.append(record)
                        
                    except Exception as e:
                        st.error(f"Error on ID {item.get('id')}: {e}")
                        results.append({
                            "id": item.get('id'),
                            "error": str(e),
                            "final_answer": None,
                            "confidence": 0.0,
                            "citations": []
                        })
                    
                    # Update Progress
                    progress_bar.progress((idx + 1) / len(questions))
                    
                    # Small delay between requests to be polite to API
                    if model_option == "Google Gemini":
                        time.sleep(2)

                status_text.text("Processing Complete!")
                st.success("✅ Evaluation Finished.")

                # --- DISPLAY RESULTS & DOWNLOAD ---
                result_jsonl = "\n".join([json.dumps(r) for r in results])
                
                st.download_button(
                    label="⬇️ Download outputs.jsonl",
                    data=result_jsonl,
                    file_name="outputs_hybrid.jsonl",
                    mime="application/json"
                )
                
                with st.expander("View Results"):
                    st.json(results)