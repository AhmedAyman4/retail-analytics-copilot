---
title: Retail Analytics Copilot
emoji: 📈
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
short_description: RAG (read docs)  SQL queries (run on SQLite)  Hybrid (both)
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 🛍️ Retail Analytics Copilot (Hybrid RAG + SQL Agent)

An autonomous AI agent built with **LangGraph** and **DSPy** that answers retail analytics questions. It intelligently switches between querying a local SQLite database (**Northwind**) and searching local documentation (**RAG**) to provide grounded, accurate answers.

[Open on Hugging Face](https://huggingface.co/spaces/ahmed-ayman/retail-analytics-copilot)

---

## 🧠 Graph Design

The agent uses a **stateful LangGraph workflow** with the following key nodes:

- **Router (Chain-of-Thought)**:  
  Classifies user intent as:

  - `SQL` (database),
  - `RAG` (policy/docs), or
  - `HYBRID` (requires both).  
    Includes **heuristic overrides** for critical keywords like `"calendar"` or `"policy"`.

- **Retriever**:  
  Uses **BM25** to search local markdown files for context (e.g., calendars, KPI definitions).

- **Planner**:  
  Analyzes retrieved context to extract specific constraints (e.g., mapping `"Summer 1997"` → date range `"1997-06-01"` to `"1997-08-31"`) **before** SQL generation.

- **SQL Generator (Strict)**:  
  A specialized **DSPy module** constrained by a _"Schema Cheat Sheet"_ to prevent hallucinations.  
  Features:

  - Regex sanitization
  - Heuristic repairs (e.g., correcting `ShipDate` → `OrderDate`)

- **Executor & Repair Loop**:  
  Runs the SQL against SQLite. If execution fails, it **feeds the error back** to the generator for up to **2 self-correction attempts**.

- **Synthesizer**:  
  Combines SQL results and text context to produce a final, typed answer **with citations**.

---

## 🚀 DSPy Optimization & Metrics

I optimized the **Router Module** using **Chain-of-Thought (CoT)** prompting.

| Metric           | Before (Simple Predict)       | After (Chain of Thought)                       |
| ---------------- | ----------------------------- | ---------------------------------------------- |
| Intent Accuracy  | 60%                           | **95%**                                        |
| Hybrid Detection | Failed often (labeled as SQL) | Successfully detects `"defined in..."` queries |
| Latency          | ~2s                           | ~4s _(worth the trade-off for accuracy)_       |

**Key Improvement**:  
The CoT Router correctly identifies that questions like _"Sales during Summer 1997"_ require a **document lookup first** (to define "Summer") before generating SQL—preventing hallucinated date ranges.

---

## ⚖️ Trade-offs & Assumptions

- **CostOfGoods Approximation**:  
  Northwind DB lacks cost data → we assume `CostOfGoods ≈ 0.7 * UnitPrice` for Gross Margin calculations (per assignment constraints).

- **Date Handling**:  
  The agent uses **string comparisons** (e.g., `>= '1997-01-01'`) instead of SQLite’s `julianday()` to reduce syntax errors with small language models.

- **Model Size vs. Speed**:  
  Using **Phi-3.5 (3.8B)** or **Llama-3.2 (3B)** on CPU → inference takes **30–60s per step**.  
  Mitigated via a **"fast path" graph**: simple queries bypass the Planner node if no context is retrieved.

---

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare the database (adds helpful views)
python add_views.py

# Run the agent on a batch of questions
python run_agent_hybrid.py \
  --batch sample_questions_hybrid_eval.jsonl \
  --out outputs_hybrid.jsonl
```
