# Deal Assistant

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/API-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Orchestration](https://img.shields.io/badge/Agent-LangGraph-7B61FF)](https://www.langchain.com/langgraph)
[![LLM](https://img.shields.io/badge/LLM-Groq-orange)](https://console.groq.com/)

A multi-turn, retrieval-grounded deal assistant that helps users find the cheapest way to pay for products and subscriptions by combining:

- offers and coupons
- cashback and price-drop logic
- credit card reward rules, category multipliers, and caps

---

## What this project does

- Searches a seeded deals dataset using scored retrieval (TF-IDF + cosine similarity)
- Compares effective prices across options
- Computes card rewards with cap-aware logic
- Streams agent responses over Server-Sent Events (SSE)
- Adds guardrails for grounding and prompt-injection defense

---

## Quick start (Windows / PowerShell)

```powershell
cd .\deal-assistant
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Then edit `.env` and add your `GROQ_API_KEY`.

Run the API:

```powershell
uvicorn app.main:app --reload
```

---

## Try it

```powershell
curl.exe -N -X POST http://127.0.0.1:8000/chat `
  -H "Content-Type: application/json" `
  -d '{\"session_id\": \"s1\", \"message\": \"What is the cheapest way to buy Netflix Premium?\"}'
```

You will receive SSE events (`token`, `tool_call`, `tool_result`, `done`) as the answer is generated.

---

## Run evaluation

```powershell
cd .\deal-assistant
python -m eval.run_eval
```

---

## Project layout

```text
deal-assistant/
├── deal-assistant/
│   ├── app/
│   ├── data/
│   ├── eval/
│   ├── requirements.txt
│   └── .env.example
└── README.md
```

---

## Notes

- The core reward calculation lives in `app/models.py` (`RewardRule.reward_for`).
- Retrieval uses a minimum score threshold to avoid low-quality matches.
- Grounding checks flag deal IDs cited in final answers that were never retrieved by tools.

