"""
Runs just the q06 demonstration case (baseline vs. improved prompt).
Separate from compare_prompts.py so re-running this doesn't burn API
quota re-doing the held-out validation, which already succeeded.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from google import genai

from rag.agent import retrieve, SYSTEM_PROMPT as BASELINE_PROMPT
from finetune.prompt_v2 import FEWSHOT_SYSTEM_PROMPT
from eval.scoring import judge

load_dotenv()

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "test_set.json")


def generate(question, context, system_template, gemini_client, model="gemini-3.5-flash-lite"):
    response = gemini_client.models.generate_content(
        model=model,
        contents=question,
        config={
            "system_instruction": system_template.format(context=context),
            "max_output_tokens": 300,
        },
    )
    time.sleep(5)
    return response.text or ""


def main():
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)
    case = next(c for c in test_set if c["id"] == "q06")

    gemini_client = genai.Client()

    question = case["question"]
    expected = case["expected_answer"]
    retrieved = retrieve(question, k=3)
    context = "\n\n---\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)

    baseline_answer = generate(question, context, BASELINE_PROMPT, gemini_client)
    improved_answer = generate(question, context, FEWSHOT_SYSTEM_PROMPT, gemini_client)

    baseline_score = judge(question, expected, baseline_answer, gemini_client)
    time.sleep(5)
    improved_score = judge(question, expected, improved_answer, gemini_client)

    result = {
        "id": "q06",
        "question": question,
        "expected_answer": expected,
        "baseline_answer": baseline_answer,
        "baseline_score": baseline_score["score"],
        "baseline_justification": baseline_score["justification"],
        "improved_answer": improved_answer,
        "improved_score": improved_score["score"],
        "improved_justification": improved_score["justification"],
    }

    out_path = os.path.join(os.path.dirname(__file__), "q06_demonstration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("=== DEMONSTRATION CASE (q06) ===")
    print(f"Question: {question}")
    print(f"Expected: {expected}")
    print()
    print(f"Baseline answer ({baseline_score['score']}): {baseline_answer}")
    print(f"  Judge: {baseline_score['justification']}")
    print()
    print(f"Improved answer ({improved_score['score']}): {improved_answer}")
    print(f"  Judge: {improved_score['justification']}")
    print()
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
