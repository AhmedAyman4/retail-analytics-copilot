import json
import dspy
import argparse
import ast
import os
from tqdm import tqdm
from agent.graph_hybrid import HybridAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--out", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    # 1. Setup DSPy with local Hugging Face Model
    # This runs locally using 'transformers'. It will download weights on first run.
    print("Loading local Hugging Face model (microsoft/Phi-3.5-mini-instruct)...")
    lm = dspy.HFModel(model='microsoft/Phi-3.5-mini-instruct')
    dspy.settings.configure(lm=lm)

    # 2. Initialize Agent
    agent_app = HybridAgent().build_graph()

    # 3. Process Batch
    results = []
    
    with open(args.batch, 'r') as f:
        lines = f.readlines()

    print(f"Processing {len(lines)} questions...")
    
    for line in tqdm(lines):
        data = json.loads(line)
        qid = data['id']
        question = data['question']
        fmt = data['format_hint']
        
        # Initial State
        initial_state = {
            "question": question,
            "format_hint": fmt,
            "retries": 0,
            "sql_error": None
        }

        # Run Graph
        try:
            output_state = agent_app.invoke(initial_state)
            
            # Post-processing answer to match format strictly (basic cleanup)
            raw_answer = output_state.get("final_answer", "")
            
            # Try to safely parse if it looks like python literal
            try:
                if fmt != "str":
                    # Remove "Final Answer:" prefix if valid
                    clean_ans = raw_answer.split("Answer:")[-1].strip()
                    parsed_answer = ast.literal_eval(clean_ans)
                else:
                    parsed_answer = raw_answer
            except:
                parsed_answer = raw_answer

            output_record = {
                "id": qid,
                "final_answer": parsed_answer,
                "sql": output_state.get("sql_query", ""),
                "confidence": 0.0, # Placeholder as allowed by constraints
                "explanation": output_state.get("explanation", ""),
                "citations": output_state.get("citations", [])
            }
            
            # Simple confidence heuristic
            if output_state.get("sql_error"):
                output_record["confidence"] = 0.1
            elif output_state.get("classification") == "sql" and output_state.get("sql_result"):
                output_record["confidence"] = 0.9
            else:
                output_record["confidence"] = 0.7

            results.append(output_record)

        except Exception as e:
            print(f"Error processing {qid}: {e}")
            results.append({
                "id": qid,
                "final_answer": None,
                "error": str(e)
            })

    # 4. Write Output
    with open(args.out, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"Done. Results written to {args.out}")

if __name__ == "__main__":
    main()