# Deal Assistant (Agentic)

A multi-turn agent that finds the cheapest way to pay for something (groceries,
subscriptions, flights, electronics) by retrieving from a seeded deals dataset
and reasoning across offers, coupons, cashback, and credit card rewards —
including caps and category multipliers.

## Stack

- **LLM**: [Groq](https://console.groq.com) — `openai/gpt-oss-20b` by default
  (fast, cheap, solid tool-calling), swap to `openai/gpt-oss-120b` in `.env`
  for stronger planning at higher latency. Groq recently retired
  `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`, so this project targets
  the current gpt-oss lineup — check `console.groq.com/docs/models` if a
  model string 404s for you.
- **Orchestration**: LangGraph — explicit `agent` → `tools` state graph,
  looped until the model stops requesting tools.
- **Retrieval**: TF-IDF + cosine similarity (scikit-learn) over the deals
  dataset, with lexical reranking and a similarity threshold for abstention.
  See "Why TF-IDF, not embeddings" below.
- **API**: FastAPI, Server-Sent Events streaming.
- **Memory**: LangGraph's `MemorySaver` checkpointer, keyed by `session_id`.

## Setup (Windows / PowerShell)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
# edit .env and paste your free Groq API key from console.groq.com/keys
```

Run the API:

```powershell
uvicorn app.main:app --reload
```

Test it:

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"session_id\": \"s1\", \"message\": \"What is the cheapest way to buy Netflix Premium?\"}'
```

You should see a stream of `data: {...}` SSE events: `token` chunks as the
answer is generated, `tool_call`/`tool_result` events showing what it looked
up, then a final `done` event with the grounding check and latency.

Run the eval harness:

```powershell
python -m eval.run_eval
```

## Project structure

```
deal-assistant/
├── data/
│   ├── deals.json          # 21 seed deals: offers, coupons, cashback, card rewards
│   └── reward_rules.json   # 4 cards: base rate, category multipliers, monthly caps
├── app/
│   ├── models.py            # Deal / RewardRule schemas, loaders, reward math
│   ├── retriever.py          # TF-IDF scored retriever, reranking, abstention
│   ├── guardrails.py          # prompt-injection sanitizer + grounding check
│   ├── tools.py                 # search_deals, compare_prices, best_card,
│   │                             # get_reward_rules, price_drop_watch
│   ├── agent.py                   # LangGraph agent, system prompt, memory
│   └── main.py                      # FastAPI app, streaming /chat endpoint
├── eval/
│   ├── labeled_set.json      # 8 labeled test cases
│   └── run_eval.py            # retrieval / accuracy / hallucination / latency
├── requirements.txt
└── .env.example
```





