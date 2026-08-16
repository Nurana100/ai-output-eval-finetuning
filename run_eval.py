"""
Runs the full evaluation: for each question in test_set.json, calls the
RAG agent, scores the answer with the LLM judge, and writes results.json
and metrics.json.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...   # or put it in a .env file
    python eval/run_eval.py
"""
import json
import os
import sys

import anthropic

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag.agent import answer as rag_answer  # noqa: E402
from eval.scoring import judge  # noqa: E402
from eval.metrics import compute_metrics  # noqa: E402

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: set ANTHROPIC_API_KEY (env var or .env file) before running.")
        sys.exit(1)

    client = anthropic.Anthropic()

    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    records = []
    for item in test_set:
        result = rag_answer(item["question"], client=client)
        judgement = judge(item["question"], item["expected_answer"], result["answer"], client=client)

        record = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "model_answer": result["answer"],
            "retrieved_sources": result["retrieved_sources"],
            "score": judgement["score"],
            "judge_justification": judgement["justification"],
            "latency_seconds": result["latency_seconds"],
            "input_tokens": result["input_tokens"],
            "output_tokens": result["output_tokens"],
            "judge_input_tokens": judgement["judge_input_tokens"],
            "judge_output_tokens": judgement["judge_output_tokens"],
        }
        records.append(record)
        print(f"[{item['id']}] score={judgement['score']}  {item['question'][:60]}")

    with open(RESULTS_PATH, "w") as f:
        json.dump(records, f, indent=2)

    metrics = compute_metrics(records)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- METRICS ---")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
