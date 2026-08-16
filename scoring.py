"""
LLM-as-judge scoring.

KNOWN LIMITATION (documented per task instructions, not just here):
Using an LLM to judge another LLM's output has known biases — it tends to
favor longer, more verbose answers, and can favor answers that are similar
in *style* to how the judge itself would write, independent of correctness.
We mitigate this partially by (a) forcing a strict rubric with a binary-ish
scale instead of open-ended praise, (b) asking the judge to justify the
score referencing specific facts, and (c) spot-checking a sample of judge
scores by hand (see report.md). We do NOT treat the judge's score as ground
truth without that manual spot-check.
"""
import json

JUDGE_SYSTEM_PROMPT = """You are grading a support assistant's answer against a reference answer.

Score from 0-2:
- 2 = Correct: all key facts in the reference are present and no facts are contradicted.
- 1 = Partially correct: some key facts present, but missing important details or containing a minor inaccuracy.
- 0 = Incorrect: contradicts the reference, hallucinates facts not in the reference, or fails to answer when the reference expects an answer (or vice versa - answers confidently when it should have declined).

Respond ONLY with JSON in this exact format, no other text:
{"score": <0, 1, or 2>, "justification": "<one sentence, citing the specific fact that was right/wrong/missing>"}
"""


def judge(question, expected_answer, model_answer, client, model="gemini-3.5-flash-lite"):
    user_prompt = f"""Question: {question}

Reference answer: {expected_answer}

Assistant's answer: {model_answer}

Score the assistant's answer."""

    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config={
            "system_instruction": JUDGE_SYSTEM_PROMPT,
            "max_output_tokens": 200,
        },
    )
    raw = (response.text or "").strip()
    # Gemini sometimes wraps JSON in ```json fences despite instructions -- strip them
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        # judge didn't follow format -- fail safe rather than crash the eval run
        parsed = {"score": None, "justification": f"UNPARSEABLE JUDGE OUTPUT: {raw}"}
    usage = response.usage_metadata
    parsed["judge_input_tokens"] = usage.prompt_token_count if usage else 0
    parsed["judge_output_tokens"] = usage.candidates_token_count if usage else 0
    return parsed
