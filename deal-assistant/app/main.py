"""
FastAPI app exposing the agent over a streaming /chat endpoint (SSE).

Run (after setting GROQ_API_KEY -- see .env.example):
    uvicorn app.main:app --reload
"""
from __future__ import annotations

import json
import time

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from pydantic import BaseModel

from app.agent import get_agent
from app.guardrails import check_grounding

app = FastAPI(title="Deal Assistant")


class ChatRequest(BaseModel):
    session_id: str
    message: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    Server-Sent Events stream. Event types sent as they happen:
      - {"type": "token", "content": "..."}                 incremental answer text
      - {"type": "tool_call", "name": "...", "input": {...}} a tool the agent invoked
      - {"type": "tool_result", "name": "...", "output": "..."} that tool's result (truncated)
      - {"type": "done", "grounding": {...}, "latency_seconds": N}  final grounding check + cost/latency
    """
    agent = get_agent()
    config = {"configurable": {"thread_id": req.session_id}}

    async def event_stream():
        start = time.time()
        final_text = ""
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                if isinstance(chunk, AIMessageChunk) and chunk.content:
                    final_text += chunk.content
                    yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"
            elif kind == "on_tool_start":
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "tool_call",
                            "name": event.get("name"),
                            "input": event["data"].get("input"),
                        }
                    )
                    + "\n\n"
                )
            elif kind == "on_tool_end":
                yield (
                    "data: "
                    + json.dumps(
                        {
                            "type": "tool_result",
                            "name": event.get("name"),
                            "output": str(event["data"].get("output"))[:500],
                        }
                    )
                    + "\n\n"
                )

        state = agent.get_state(config)
        messages = state.values.get("messages", [])
        grounding = check_grounding(final_text, messages)
        latency = round(time.time() - start, 2)
        yield (
            "data: "
            + json.dumps({"type": "done", "grounding": grounding, "latency_seconds": latency})
            + "\n\n"
        )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
