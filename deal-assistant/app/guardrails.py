"""
Guardrails: two independent mechanisms.

1. sanitize_text() — deal descriptions are untrusted data (anyone could have
   written an instruction-like payload into the dataset). Before a
   description is handed back to the LLM inside a tool result, strip known
   injection trigger phrases so they can't be read as commands.

2. check_grounding() — after the agent answers, verify every deal id it
   cited actually appeared in a tool result during this run. This is the
   mechanical backbone of "never invent a deal": the system prompt asks the
   model not to, this function checks that it didn't.
"""
from __future__ import annotations

import re

from langchain_core.messages import ToolMessage

DEAL_ID_RE = re.compile(r"\bD\d{3}\b")

_INJECTION_PATTERNS = [
    re.compile(r"ignore (all|any|previous|prior)[^.]*instructions", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"you are now[^.]*", re.I),
    re.compile(r"disregard (the )?(above|previous)[^.]*", re.I),
    re.compile(r"reveal (your|the)[^.]*(system|instructions)[^.]*", re.I),
    re.compile(r"developer mode", re.I),
]


def sanitize_text(text: str) -> str:
    """Neutralize likely prompt-injection payloads embedded in deal text."""
    cleaned = text
    for pat in _INJECTION_PATTERNS:
        cleaned = pat.sub("[redacted: instruction-like text ignored]", cleaned)
    return cleaned


def extract_grounded_ids(messages) -> set[str]:
    """Every deal id that actually appeared in a ToolMessage this run —
    the only ids the final answer is allowed to cite."""
    grounded: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            grounded.update(DEAL_ID_RE.findall(m.content))
    return grounded


def check_grounding(final_answer: str, messages) -> dict:
    grounded_ids = extract_grounded_ids(messages)
    cited_ids = set(DEAL_ID_RE.findall(final_answer or ""))
    ungrounded = cited_ids - grounded_ids
    return {
        "cited_ids": sorted(cited_ids),
        "grounded_ids_available": sorted(grounded_ids),
        "ungrounded_ids": sorted(ungrounded),
        "hallucination_flag": len(ungrounded) > 0,
    }
