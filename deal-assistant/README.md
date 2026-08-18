# Deal Assistant

A multi-turn agent that finds the cheapest way to pay for things (groceries, subscriptions, flights, electronics) by retrieving from a seeded deals dataset and reasoning across offers, coupons, cashback, and credit card rewards — including caps and category multipliers.

This README was updated to include architecture, setup, benchmarking results, and usage examples.

---

## Quick Start (Windows / PowerShell)

1. Open PowerShell and change to the project folder:

```powershell
cd C:/Users/Shastry's/Downloads/deal-assistant/deal-assistant
```

2. Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Copy and set your Groq API key (replace the placeholder with your key):

```powershell
copy .env.example .env
# then open .env and set GROQ_API_KEY to your key
```

4. Run the API server:

```powershell
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
├── data/
│   ├── deals.json          # seed deals: offers, coupons, cashback, card rewards
│   └── reward_rules.json   # cards: base rates, multipliers, monthly caps
├── app/
│   ├── models.py           # Deal / RewardRule schemas, reward math
│   ├── retriever.py        # TF-IDF retriever + reranking + abstention
│   ├── guardrails.py       # prompt-injection sanitizer + grounding check
│   ├── tools.py            # search_deals, compare_prices, best_card, price_drop_watch
│   ├── agent.py            # LangGraph agent, system prompt, memory
│   └── main.py             # FastAPI app, streaming /chat endpoint
├── eval/
│   ├── labeled_set.json    # labeled test cases for eval harness
│   └── run_eval.py         # retrieval / accuracy / hallucination / latency
├── requirements.txt
└── .env.example
```

---

## Architecture

- LLM: Groq (`openai/gpt-oss-20b` by default) — used for planning, tool-calling, and streaming responses.
- Orchestration: LangGraph — agent graph with explicit tools; the loop runs until the model stops requesting tools.
- Retrieval: TF-IDF (scikit-learn) + cosine similarity over `data/deals.json` with lexical reranking and an abstention threshold.
- API: FastAPI with Server-Sent Events (SSE) for streaming tokens, tool_call and tool_result events.
- Memory: LangGraph `MemorySaver` checkpointer keyed by `session_id` for multi-turn sessions.
- Guardrails: mechanical grounding/hallucination checks (deal id whitelist), and prompt-injection sanitization on retrieved text.

Design note: reward math is centralized in `RewardRule.reward_for(amount, category)` in `app/models.py` so all reward numbers are deterministic and auditable.

---

## Benchmarks (latest run)

These results were produced by the included eval harness (`python -m eval.run_eval`) on this machine.

- retrieval_precision_avg: 0.59
- retrieval_recall_avg: 0.80
- answer_accuracy: 0.71
- hallucination_rate: 0.0
- avg_latency_seconds: 18.27

Full JSON report saved at: `eval/eval_results.json`

Notes on the numbers:
- Precision/recall are measured on the small labeled set (8 cases) in `eval/labeled_set.json` and reflect retrieval + reranking behaviour.
- Latency is the average end-to-end time per eval case (includes LLM inference over the Groq API) on the machine where the harness was run.

---

## Security & credentials

- The project expects a `GROQ_API_KEY` in `.env`. Do not commit your real API key to git.
- After local benchmarking or demos, remove or rotate any keys written to `.env`.

To remove the key from `.env`:

```powershell
Set-Content -Path .env -Value "GROQ_API_KEY=your_groq_api_key_here`nGROQ_MODEL=openai/gpt-oss-20b"
```

---

## Running the eval harness (benchmark)

```powershell
# With venv active
python -m eval.run_eval
# Output: prints summary and writes eval/eval_results.json
```

To run multiple iterations (measure variance), script the command and clear any caches between runs.

---

## Example usage (single run)

1. Start server (keep running):

```powershell
uvicorn app.main:app --reload
```

2. Send a POST to /chat:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" -d "{\"session_id\":\"run1\",\"message\":\"What is the cheapest way to buy Netflix Premium?\"}"
```

You will receive SSE events containing `token`, `tool_call`, `tool_result`, and a final `done` event that includes a grounding check and latency.

---

## Development notes

- The retriever uses a `min_score` abstention threshold to avoid low-relevance padding; when no reliable deal is found tools return `{"status": "no_reliable_deal_found"}` and the agent is required to say so.
- `deals.json` includes a deliberately poisoned record to exercise prompt-injection defenses; `guardrails.sanitize_text()` strips known trigger phrases.

---

## Where to look next

- `app/models.py` — reward math & schemas
- `app/retriever.py` — TF-IDF vectorizer, reranking, abstention
- `app/agent.py` — the LangGraph agent and system prompt
- `app/main.py` — FastAPI endpoint and SSE streaming
- `eval/run_eval.py` — harness that produced the benchmark above

---

If you want, I can:

- Push this README update to your repository (requires git push access).
- Remove the API key from `.env` for security.
- Create a small script to run the server, execute a few test queries, and record per-query latencies to a CSV.

---

(last updated: benchmark run results included)
