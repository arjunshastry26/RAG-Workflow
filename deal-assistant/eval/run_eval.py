"""
Eval harness: labeled set -> retrieval quality, answer accuracy,
hallucination rate, latency.

Retrieval metrics run directly against the retriever and need no LLM --
they work fully offline. Answer accuracy, hallucination rate, and latency
run the full agent, so they need a live GROQ_API_KEY in your environment.

Usage:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from app.guardrails import check_grounding
from app.retriever import get_retriever

EVAL_DIR = Path(__file__).resolve().parent


def load_cases() -> list[dict]:
    with open(EVAL_DIR / "labeled_set.json", encoding="utf-8") as f:
        return json.load(f)


def retrieval_metrics(query: str, expected_ids: list[str]) -> dict:
    retriever = get_retriever()
    results, weak = retriever.search(query, top_k=10)
    retrieved_ids = {d.id for d, _ in results}
    expected = set(expected_ids)
    if expected:
        tp = len(retrieved_ids & expected)
        precision = tp / len(retrieved_ids) if retrieved_ids else 0.0
        recall = tp / len(expected)
    else:
        # nothing should have matched -- correct behaviour is to abstain
        precision = recall = 1.0 if weak else 0.0
    return {"precision": round(precision, 2), "recall": round(recall, 2), "weak": weak}


def run_eval(use_llm: bool = True) -> dict:
    from app.agent import run_once  # imported lazily so retrieval-only runs don't need GROQ_API_KEY

    cases = load_cases()
    report: dict = {"cases": [], "summary": {}}
    precisions, recalls, accurate, hallucinated, latencies = [], [], [], [], []

    for case in cases:
        row: dict = {"id": case["id"], "query": case["query"]}

        if case["type"] in ("search_deals", "compare_prices", "multi_tool", "prompt_injection"):
            rm = retrieval_metrics(case["query"], case.get("expected_deal_ids", []))
            row["retrieval"] = rm
            precisions.append(rm["precision"])
            recalls.append(rm["recall"])

        if use_llm:
            thread_id = case.get("depends_on", case["id"])
            start = time.time()
            try:
                answer, messages = run_once(case["query"], thread_id=thread_id)
            except Exception as e:  # e.g. no/invalid GROQ_API_KEY, or offline
                row["error"] = str(e)
                report["cases"].append(row)
                continue
            latency = round(time.time() - start, 2)
            latencies.append(latency)
            row["answer"] = answer
            row["latency_seconds"] = latency

            grounding = check_grounding(answer, messages)
            row["grounding"] = grounding
            hallucinated.append(grounding["hallucination_flag"])

            if case.get("should_abstain"):
                is_correct = any(
                    phrase in answer.lower()
                    for phrase in ("no reliable deal", "couldn't find", "could not find", "unable to find")
                )
            elif "expected_answer_contains" in case:
                is_correct = all(kw.lower() in answer.lower() for kw in case["expected_answer_contains"])
            elif "expected_deal_ids" in case:
                is_correct = any(d_id in answer for d_id in case["expected_deal_ids"])
            else:
                is_correct = True

            if "must_not_contain" in case:
                is_correct = is_correct and not any(
                    bad.lower() in answer.lower() for bad in case["must_not_contain"]
                )

            row["answer_correct"] = is_correct
            accurate.append(is_correct)

        report["cases"].append(row)

    if precisions:
        report["summary"]["retrieval_precision_avg"] = round(sum(precisions) / len(precisions), 2)
        report["summary"]["retrieval_recall_avg"] = round(sum(recalls) / len(recalls), 2)
    if use_llm and accurate:
        report["summary"]["answer_accuracy"] = round(sum(accurate) / len(accurate), 2)
        report["summary"]["hallucination_rate"] = round(sum(hallucinated) / len(hallucinated), 2)
        report["summary"]["avg_latency_seconds"] = (
            round(sum(latencies) / len(latencies), 2) if latencies else None
        )

    return report


if __name__ == "__main__":
    result = run_eval(use_llm=True)
    print(json.dumps(result["summary"], indent=2))
    with open(EVAL_DIR / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull report written to {EVAL_DIR / 'eval_results.json'}")
