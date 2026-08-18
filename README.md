# Deal Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Orchestration](https://img.shields.io/badge/Agent-LangGraph-7B61FF)](https://www.langchain.com/langgraph)
[![LLM](https://img.shields.io/badge/LLM-Groq-orange)](https://console.groq.com/)

A multi-turn, retrieval-grounded deal assistant that finds the cheapest way to pay for products and subscriptions by reasoning across offers, coupons, cashback, and credit-card rewards — including caps and category multipliers.

This README contains setup, architecture, benchmarking results, and examples for running the project locally.

---

## Quick Start (Windows / PowerShell)

1. Open PowerShell and change to the project folder:

```powershell
cd C:/Users/Shastry's/Downloads/deal-assistant
```

2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r deal-assistant/requirements.txt
```

3. Copy and set your Groq API key (replace the placeholder with your key):

```powershell
copy deal-assistant/.env.example deal-assistant/.env
# then open deal-assistant/.env and set GROQ_API_KEY to your key
```

4. Run the API server:

```powershell
cd deal-assistant
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

5. Test with a single request (PowerShell):

```powershell
$body = @{session_id='s1'; message='What is the cheapest way to buy Netflix Premium?'} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType 'application/json' -Body $body
```

Or using curl.exe (PowerShell-escaped JSON):

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"session_id\":\"s1\",\"message\":\"What is the cheapest way to buy Netflix Premium?\"}"
```

---

## Project structure

```
deal-assistant/
├── deal-assistant/                # application and evaluation harness
│   ├── app/                       # FastAPI app, agent, tools, retriever
│   ├── data/                      # deals.json, reward_rules.json
│   ├── eval/                      # eval harness and labeled test cases
│   ├── requirements.txt
│   └── .env.example
├── README.md                      # <- this file (repository root)
```

---

## Architecture

- LLM: Groq (`openai/gpt-oss-20b` by default) — used for planning, tool-calling, and streaming responses.
- Orchestration: LangGraph — explicit agent -> tools graph that loops until the model stops requesting tools.
- Retrieval: TF-IDF (scikit-learn) + cosine similarity over `deal-assistant/data/deals.json` with lexical reranking and an abstention threshold to avoid low-quality padding.
- API: FastAPI with Server-Sent Events (SSE) for streaming tokens plus tool_call/tool_result events.
- Memory: LangGraph's `MemorySaver` checkpointer keyed by `session_id` for multi-turn sessions.
- Guardrails: mechanical grounding/hallucination checks (deal id whitelist) and prompt-injection sanitization on retrieved text.

Design note: reward math is centralized in `deal-assistant/app/models.py` (`RewardRule.reward_for(amount, category)`), making reward calculations deterministic and auditable.

---

## Benchmarks (latest run)

These results were produced by the included eval harness (`python -m eval.run_eval`) run locally on this machine.

- retrieval_precision_avg: 0.59
- retrieval_recall_avg: 0.80
- answer_accuracy: 0.71
- hallucination_rate: 0.0
- avg_latency_seconds: 18.27

Full JSON report saved at: `deal-assistant/eval/eval_results.json`

Notes:
- Precision/recall are measured on a small labeled set of 8 cases (`deal-assistant/eval/labeled_set.json`).
- Latency is end-to-end and includes LLM inference time against the Groq API on this machine.

---

## Security & credentials

- Put your Groq API key in `deal-assistant/.env` as `GROQ_API_KEY` — do not commit your key to git.
- After demos or benchmarking, remove or rotate any keys stored in `.env`.

To reset `.env` to the placeholder:

```powershell
Set-Content -Path deal-assistant/.env -Value "GROQ_API_KEY=your_groq_api_key_here`nGROQ_MODEL=openai/gpt-oss-20b"
```

---

## Running the eval harness (benchmark)

```powershell
# from repository root
cd deal-assistant
python -m eval.run_eval
# prints summary and writes eval/eval_results.json
```

To run multiple iterations for variance measurement, script the command and optionally clear any caches between runs.

---

## Example usage (single run)

1. Start server:

```powershell
cd deal-assistant
uvicorn app.main:app --reload
```

2. Send a POST to /chat:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"session_id\":\"run1\",\"message\":\"What is the cheapest way to buy Netflix Premium?\"}"
```

You will receive SSE events containing `token`, `tool_call`, `tool_result`, and a final `done` event that includes a grounding check and latency.

---

## Development notes

- Retriever uses a `min_score` abstention threshold to avoid low-relevance padding; when nothing reliable is found tools return `{"status": "no_reliable_deal_found"}` and the agent is required to say so.
- `deal-assistant/data/deals.json` includes a deliberately poisoned record to exercise prompt-injection defenses; `deal-assistant/app/guardrails.py` sanitizes retrieved text.

---

## Where to look next

- `deal-assistant/app/models.py` — reward math & schemas
- `deal-assistant/app/retriever.py` — TF-IDF vectorizer, reranking, abstention
- `deal-assistant/app/agent.py` — the LangGraph agent and system prompt
- `deal-assistant/app/main.py` — FastAPI endpoint and SSE streaming
- `deal-assistant/eval/run_eval.py` — harness that produced the benchmark above

---

If you'd like, I can also:
- Remove the API key from `deal-assistant/.env` now for security.
- Add an architecture diagram file (SVG/PNG) or a separate docs/ folder.
- Create an automated benchmarking script that runs N iterations and writes CSV results.

(last updated: benchmark run results included)
