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

Mentor feedback on the original version of this script: a 4-question,
1-trial-each held-out set produced baseline == improved (2.00 == 2.00) on
every question, which only shows "no regression," not "measurable
improvement" -- and per failure_analysis.md's own note on generation
non-determinism, a single sample per question isn't reliable evidence
either way. This version addresses both: the held-out set is expanded to
7 questions, and each (question, prompt version) pair is run N_TRIALS
times and averaged, so the comparison is less sensitive to one sample's
random variance.
"""
import json
import os
import statistics

from dotenv import load_dotenv
from google import genai

from agent import retrieve, SYSTEM_PROMPT as BASELINE_PROMPT
from prompt_v2 import FEWSHOT_SYSTEM_PROMPT
from scoring import judge

load_dotenv()

TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

# Held out on purpose: does NOT include q06 (the diagnosed sample) or q04
# (a *retrieval* failure -- out of scope for this prompt-level fix).
# Expanded from the original 4 to 7 questions: every remaining "normal"
# question whose reference answer contains a qualifier/exception clause
# (plan tier, timing rule, or scope limit), the same failure shape q06
# needed fixed, but none examined while designing the fix:
#   q05 — full-text search plan-tier qualifier (Free vs. Pro/Team)
#   q09 — refund guarantee's scope limit ("does not apply to renewals...")
#   q10 — Locked Notes: staff-access + unrecoverable-password qualifiers
#   q11 — account-deletion retention-window qualifier (90 days, reversible)
#   q12 — downgrade timing qualifier ("next billing cycle, not immediately")
#   q13 — attachment size limit qualifier (Pro-plan specific)
#   q16 — collaboration eligibility qualifier (Team-only, no Free/Pro path)
HELD_OUT_IDS = {"q05", "q09", "q10", "q11", "q12", "q13", "q16"}

# q06 is the diagnosed sample the fix targets directly. It is run and
# reported SEPARATELY from the held-out set above, as a demonstration of
# the fix rather than proof it generalizes -- the held-out set is what
# establishes that.
DEMONSTRATION_ID = "q06"

# Run each (question, prompt version) pair this many times and average the
# judge score, instead of trusting a single noisy sample.
N_TRIALS = 3


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


def run_trials(question, expected, context, system_template, gemini_client, n=N_TRIALS):
    """Runs `n` independent generate+judge trials and returns each one, so
    callers can look at the spread across trials, not just the mean."""
    trials = []
    for _ in range(n):
        answer = generate(question, context, system_template, gemini_client)
        result = judge(question, expected, answer, gemini_client)
        trials.append({"answer": answer, "score": result["score"], "justification": result["justification"]})
    return trials


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

        baseline_trials = run_trials(question, expected, context, BASELINE_PROMPT, gemini_client)
        improved_trials = run_trials(question, expected, context, FEWSHOT_SYSTEM_PROMPT, gemini_client)

        baseline_scores = [t["score"] for t in baseline_trials if t["score"] is not None]
        improved_scores = [t["score"] for t in improved_trials if t["score"] is not None]

        return {
            "id": case["id"],
            "question": question,
            "expected_answer": expected,
            "baseline_trials": baseline_trials,
            "improved_trials": improved_trials,
            "baseline_avg": (sum(baseline_scores) / len(baseline_scores)) if baseline_scores else None,
            "improved_avg": (sum(improved_scores) / len(improved_scores)) if improved_scores else None,
        }

    results = [run_case(c) for c in cases]
    demo_result = run_case(demo_case) if demo_case else None

    out_path = os.path.join(os.path.dirname(__file__), "prompt_comparison_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"held_out": results, "demonstration_q06": demo_result, "n_trials": N_TRIALS}, f, indent=2)

    if demo_result:
        print(f"=== DEMONSTRATION CASE (q06, {N_TRIALS} trials/version — not used for validation proof) ===")
        print(f"Baseline avg: {demo_result['baseline_avg']:.2f}  scores={[t['score'] for t in demo_result['baseline_trials']]}")
        print(f"Improved avg: {demo_result['improved_avg']:.2f}  scores={[t['score'] for t in demo_result['improved_trials']]}")
        print()

    print(f"=== HELD-OUT VALIDATION ({len(results)} questions x {N_TRIALS} trials each, not used to design the fix) ===")
    print(f"{'ID':<6}{'Baseline avg':<14}{'Improved avg':<14}{'Question'}")
    for r in results:
        b = f"{r['baseline_avg']:.2f}" if r["baseline_avg"] is not None else "n/a"
        i = f"{r['improved_avg']:.2f}" if r["improved_avg"] is not None else "n/a"
        print(f"{r['id']:<6}{b:<14}{i:<14}{r['question'][:55]}")

    valid = [r for r in results if r["baseline_avg"] is not None and r["improved_avg"] is not None]
    if len(valid) < len(results):
        print(f"\nWARNING: {len(results) - len(valid)} case(s) had unparseable judge scores in every trial and were excluded.")
    if valid:
        overall_baseline = sum(r["baseline_avg"] for r in valid) / len(valid)
        overall_improved = sum(r["improved_avg"] for r in valid) / len(valid)
        n_improved = sum(1 for r in valid if r["improved_avg"] > r["baseline_avg"])
        n_regressed = sum(1 for r in valid if r["improved_avg"] < r["baseline_avg"])
        n_tied = len(valid) - n_improved - n_regressed
        print(f"\nOverall held-out avg — baseline: {overall_baseline:.2f}  improved: {overall_improved:.2f}")
        print(f"Per-question direction: {n_improved} improved, {n_regressed} regressed, {n_tied} tied")
        if len(valid) > 1:
            stdev_b = statistics.pstdev(r["baseline_avg"] for r in valid)
            stdev_i = statistics.pstdev(r["improved_avg"] for r in valid)
            print(f"Spread across questions (population stdev of per-question avg) — baseline: {stdev_b:.2f}  improved: {stdev_i:.2f}")
    print(f"Full detail written to {out_path}")


if __name__ == "__main__":
    main()
