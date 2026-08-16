"""
Checkpoint 5 — before/after comparison for the prompt fix targeting the
q06 failure category (model drops a qualifying detail that was present in
its retrieved context).

Anti-pollution note: q06 was the sample used to *diagnose* this failure
and design the fix (the few-shot example in prompt_v2.py is written from
general principle, not fitted to q06's specific wording). To avoid
cyclical validation, q06 is EXCLUDED from the held-out set this script
scores on. Validation runs on other questions the fix was not designed
against — chosen because their reference answers also contain a
qualifying detail (plan tier, limit, or exception) that a model could
plausibly drop, same failure shape as q06 but different content.
"""
import json
import os

from dotenv import load_dotenv
from google import genai

from agent import retrieve, SYSTEM_PROMPT as BASELINE_PROMPT
from prompt_v2 import FEWSHOT_SYSTEM_PROMPT
from scoring import judge

load_dotenv()

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

# Held out on purpose: does NOT include q06 (the diagnosed sample).
# Picked because each reference answer contains a qualifier/exception
# clause (plan-tier restriction, timing rule, or scope limit) similar in
# shape to what q06 needed, but not examined while designing the fix:
#   q09 — refund guarantee's scope limit ("does not apply to renewals...")
#   q10 — Locked Notes: staff-access + unrecoverable-password qualifiers
#   q12 — downgrade timing qualifier ("next billing cycle, not immediately")
#   q16 — collaboration eligibility qualifier ("Team-only, no Free/Pro path")
HELD_OUT_IDS = {"q09", "q10", "q12", "q16"}

# q06 is the diagnosed sample the fix targets directly. It is run and
# reported SEPARATELY from the held-out set above, as a demonstration of
# the fix rather than proof it generalizes -- the held-out set is what
# establishes that.
DEMONSTRATION_ID = "q06"


def generate(question, context, system_template, gemini_client, model="gemini-3.5-flash-lite"):
    response = gemini_client.models.generate_content(
        model=model,
        contents=question,
        config={
            "system_instruction": system_template.format(context=context),
            "max_output_tokens": 300,
        },
    )
    return response.text or ""


def main():
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        test_set = json.load(f)

    cases = [c for c in test_set if c["id"] in HELD_OUT_IDS]
    demo_case = next((c for c in test_set if c["id"] == DEMONSTRATION_ID), None)
    if len(cases) != len(HELD_OUT_IDS):
        found = {c["id"] for c in cases}
        missing = HELD_OUT_IDS - found
        print(f"WARNING: could not find held-out ids in test_set.json: {missing}")
        print("Edit HELD_OUT_IDS in this script to match your actual test_set.json ids.")

    gemini_client = genai.Client()

    def run_case(case):
        question = case["question"]
        expected = case["expected_answer"]
        retrieved = retrieve(question, k=3)
        context = "\n\n---\n\n".join(f"[{r['source']}]\n{r['text']}" for r in retrieved)

        baseline_answer = generate(question, context, BASELINE_PROMPT, gemini_client)
        improved_answer = generate(question, context, FEWSHOT_SYSTEM_PROMPT, gemini_client)

        baseline_score = judge(question, expected, baseline_answer, gemini_client)
        improved_score = judge(question, expected, improved_answer, gemini_client)

        return {
            "id": case["id"],
            "question": question,
            "expected_answer": expected,
            "baseline_answer": baseline_answer,
            "baseline_score": baseline_score["score"],
            "baseline_justification": baseline_score["justification"],
            "improved_answer": improved_answer,
            "improved_score": improved_score["score"],
            "improved_justification": improved_score["justification"],
        }

    results = [run_case(c) for c in cases]

    demo_result = run_case(demo_case) if demo_case else None

    out_path = os.path.join(os.path.dirname(__file__), "prompt_comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"held_out": results, "demonstration_q06": demo_result}, f, indent=2)

    if demo_result:
        print("=== DEMONSTRATION CASE (q06 — the diagnosed sample, not used for validation proof) ===")
        print(f"Baseline ({demo_result['baseline_score']}): {demo_result['baseline_answer']}")
        print(f"  Judge: {demo_result['baseline_justification']}")
        print(f"Improved ({demo_result['improved_score']}): {demo_result['improved_answer']}")
        print(f"  Judge: {demo_result['improved_justification']}")
        print()

    print("=== HELD-OUT VALIDATION (not used to design the fix) ===")
    print(f"{'ID':<6}{'Baseline':<10}{'Improved':<10}{'Question'}")
    for r in results:
        print(f"{r['id']:<6}{str(r['baseline_score']):<10}{str(r['improved_score']):<10}{r['question'][:60]}")

    valid = [r for r in results if r["baseline_score"] is not None and r["improved_score"] is not None]
    if len(valid) < len(results):
        print(f"\nWARNING: {len(results) - len(valid)} case(s) had an unparseable judge score and were excluded from the average.")
    if valid:
        avg_baseline = sum(r["baseline_score"] for r in valid) / len(valid)
        avg_improved = sum(r["improved_score"] for r in valid) / len(valid)
        print(f"\nHeld-out avg score — baseline: {avg_baseline:.2f}  improved: {avg_improved:.2f}")
    print(f"Full detail written to {out_path}")


if __name__ == "__main__":
    main()
