"""
Runs the full evaluation: for each question in test_set.json, calls the
RAG agent, scores the answer with the LLM judge, and writes results.json
and metrics.json.

Usage:
    export GEMINI_API_KEY=...   # or put it in a .env file
    python run_eval.py

Note: the free tier of the Gemini API is rate-limited (as of writing,
gemini-3.5-flash-lite allows 15 requests/minute). Each question makes 2
calls (agent answer + judge), so this script paces itself and retries
automatically on 429 (rate limit) errors rather than crashing.
"""
import json
import os
import sys
import time

from google import genai
from google.genai import errors as genai_errors

from agent import answer as rag_answer
from scoring import judge
from metrics import compute_metrics

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")
METRICS_PATH = os.path.join(os.path.dirname(__file__), "metrics.json")

SECONDS_BETWEEN_CALLS = 5  # paces us to ~12 requests/minute, under the 15/min free-tier cap
MAX_RETRIES = 5


def call_with_retry(fn, *args, **kwargs):
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except genai_errors.ClientError as e:
            if getattr(e, "code", None) == 429 and attempt < MAX_RETRIES - 1:
                wait = 15 * (attempt + 1)
                print(f"    rate limited, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: set GEMINI_API_KEY (env var or .env file) before running.")
        sys.exit(1)

    client = genai.Client()

    with open(TEST_SET_PATH) as f:
        test_set = json.load(f)

    records = []
    for item in test_set:
        result = call_with_retry(rag_answer, item["question"], client=client)
        time.sleep(SECONDS_BETWEEN_CALLS)
        judgement = call_with_retry(
            judge, item["question"], item["expected_answer"], result["answer"], client=client
        )

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

        # save progress after every question so a crash doesn't lose completed work
        with open(RESULTS_PATH, "w") as f:
            json.dump(records, f, indent=2)

        time.sleep(SECONDS_BETWEEN_CALLS)

    metrics = compute_metrics(records)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- METRICS ---")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
