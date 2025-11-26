import json
import dspy
import argparse
import ast
import os
import sys
import time
import requests

sys.path.append(os.getcwd())
from tqdm import tqdm
from agent.graph_hybrid import HybridAgent

def wait_for_server():
    """Waits for Ollama server to be ready."""
    print("Waiting for Ollama (localhost:11434)...")
    for i in range(30):
        try:
            requests.get("http://localhost:11434")
            print("Server is ready.")
            return True
        except:
            time.sleep(2)
            print(".", end="", flush=True)
    print("\nServer check timed out. Proceeding anyway...")
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    wait_for_server()

    # Configure DSPy for Ollama
    print("Configuring DSPy for Ollama (phi3.5:3.8b)...")
    lm = dspy.LM(
        model="ollama_chat/phi3.5:3.8b",
        api_base="http://localhost:11434",
        api_key=""
    )
    dspy.configure(lm=lm)

    # Initialize Agent
    agent_app = HybridAgent().build_graph()

    results = []
    with open(args.batch, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} questions...")
    
    for line in tqdm(lines):
        data = json.loads(line)
        qid = data['id']
        question = data['question']
        fmt = data['format_hint']
        
        initial_state = {
            "question": question,
            "format_hint": fmt,
            "retries": 0,
            "sql_error": None
        }

        try:
            output = agent_app.invoke(initial_state)
            
            raw_answer = output.get("final_answer", "")
            try:
                if fmt != "str":
                    clean = str(raw_answer).split("Answer:")[-1].strip()
                    parsed = ast.literal_eval(clean)
                else:
                    parsed = raw_answer
            except:
                parsed = raw_answer

            res = {
                "id": qid,
                "final_answer": parsed,
                "sql": output.get("sql_query", ""),
                "explanation": output.get("explanation", ""),
                "citations": output.get("citations", [])
            }
            results.append(res)

        except Exception as e:
            print(f"Error {qid}: {e}")
            results.append({"id": qid, "error": str(e)})

    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Results written to {args.out}")

if __name__ == "__main__":
    main()