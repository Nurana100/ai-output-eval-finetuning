# Evaluation Report — Nimbus Notes RAG Agent

## 1. Objective

Build a framework to test and evaluate the Nimbus Notes RAG support agent
against a curated test set, identify real failure cases with root-cause
analysis, and apply a targeted fix to one identified failure category
with a before/after comparison.

## 2. Methodology

### 2.1 Test Set

18 question/expected-answer pairs (`test_set.json`), covering all 5
knowledge-base documents:
- 13 normal questions, one or more per document (pricing, features,
  troubleshooting, account management, privacy/security).
- 5 outliers: an informally-phrased paraphrase (q14), two out-of-scope
  questions (q15, q18), one deliberately ambiguous question (q16), and
  one trap question conflating two similar policies (q17).

### 2.2 System Under Test

The RAG agent (`agent.py`) retrieves context with a TF-IDF + FAISS
index (`build_index.py`) over the knowledge base, then generates an
answer with `gemini-3.5-flash-lite`, constrained to a 2-4 sentence,
context-only response.

### 2.3 Scoring

An LLM-as-judge (`scoring.py`, also `gemini-3.5-flash-lite`) scores
each answer 0-2 against the reference answer:
- 2 = correct, all key facts present, nothing contradicted
- 1 = partially correct, missing details or a minor inaccuracy
- 0 = incorrect, contradicts the reference, hallucinates, or
  wrongly answers/declines relative to what the reference expects

### 2.4 Metrics

`run_eval.py` runs the full suite and reports pass rate (strict:
score==2, lenient: score>=1), average latency, and average token cost
(`metrics.py`), using current published pricing for
`gemini-3.5-flash-lite`.

## 3. Results

### 3.1 Aggregate Metrics (Checkpoint 3)

| Metric | Value |
|---|---|
| Pass rate (strict, score==2) | 0.889 |
| Pass rate (lenient, score>=1) | 0.944 |
| Avg latency | 0.983s / question |
| Avg cost | $0.00036 / question ($0.0065 total, 18 questions) |

### 3.2 Failure Analysis (Checkpoint 4)

Manual review of the full result set (not just the automated pass/fail
flags) surfaced three distinct failure cases, each with a different root
cause — full detail in `failure_analysis.md`:

| ID | Question | Score | Root Cause |
|---|---|---|---|
| q04 | Offline use without internet | 0 | **Retrieval** — TF-IDF vocabulary mismatch. The question's wording ("without an internet connection") shares no vocabulary with the answer's wording ("offline mode"), so an irrelevant chunk that shares literal words outranks the correct one. The correct chunk did not appear in the top 10 results for this query. |
| q06 | Export formats + bulk export restriction | 1 | **Generation** — the correct chunk, containing the full answer, was retrieved and in context; the model still dropped the Pro/Team qualifier when generating a concise answer. |
| q17 | Mid-cycle cancellation partial refund (trap) | 2 (judge) | **Judge-scoring bias** — the model's answer was factually correct but never engaged with the distractor policy (14-day money-back guarantee) the question was designed to test. The judge scored it full credit anyway, since its rubric checks final-answer correctness, not demonstrated disambiguation. |

An earlier draft of this analysis initially mis-classified q06 as a
retrieval failure; a deeper diagnostic (printing full retrieved-chunk
text rather than just source filenames) corrected this. That correction
is documented in `failure_analysis.md` for transparency.

### 3.3 Fix and Validation (Checkpoint 5)

**Fix:** A few-shot prompt (`prompt_v2.py`) targeting the q06
failure category (generation dropping qualifiers under a brevity
constraint). Adds an explicit instruction plus one worked example showing
that plan-tier/limit/exception qualifiers must be preserved even in a
short answer.

**Held-out validation** (`compare_prompts.py`): to avoid
cyclical validation (testing a fix on the same sample used to design it),
the fix was scored against questions *not* used during development —
chosen because their reference answers also contain a qualifier of the
same shape (plan-tier restriction, timing rule, or scope limit) q06
needed.

**Revised per mentor review (8/17/2026):** the original run used 4
held-out questions (q09, q10, q12, q16) at 1 trial each, and both
baseline and improved tied at 2.00 avg — which only shows "no
regression," not "measurable improvement," and (per the non-determinism
finding below) a single sample per question isn't strong evidence either
way. `compare_prompts.py` now uses 7 held-out questions (q05, q09, q10,
q11, q12, q13, q16) and averages 3 trials per question per prompt
version.

| Set | Baseline avg score | Improved avg score |
|---|---|---|
| Held-out, original (q09, q10, q12, q16 — 1 trial each) | 2.00 | 2.00 |
| Held-out, revised (7 questions x 3 trials each) | *(re-run `compare_prompts.py` and paste numbers here)* | *(re-run `compare_prompts.py` and paste numbers here)* |

Original result: no regressions, but all four held-out questions already
scored full credit before the change, so that run demonstrated the fix
is safe rather than demonstrating a score improvement. The revised
methodology (larger set, multi-trial averaging) is what should be used
to decide whether the fix shows a real, measurable improvement — run it
with a live `GEMINI_API_KEY` and record the outcome above before treating
this checkpoint as fully validated.

**Direct q06 comparison** (`run_q06_demo.py`, reported
separately from the held-out set since q06 was the diagnosed sample, not
a validation sample): both baseline and improved prompts scored 2 on this
run. This surfaced an important limitation — **LLM generation
non-determinism**: the same unmodified baseline prompt, on the same
question, scored 1 in the original Checkpoint 3 eval run (dropping the
qualifier) and 2 on this later run (including it), with no code changes
between the two. A single before/after sample is therefore not reliable
evidence a fix works or doesn't; the held-out multi-question aggregate
is the more trustworthy signal. Full detail and reasoning in
`failure_analysis.md`.

## 4. Known Limitations

- **LLM-as-judge bias.** The judge is itself an LLM and shares known
  biases — leniency, and reward for answers that hit the reference's key
  facts without checking *how* the model arrived at them. q17 is a
  concrete example: a factually correct but reasoning-incomplete answer
  was scored full credit. Aggregate pass-rate metrics should be read as
  an upper bound, especially on trap/discriminative questions, not as
  ground truth. Manual spot-checking of outlier cases remains necessary.
- **Generation non-determinism.** Sampling variance means single-run
  before/after comparisons on one question are weak evidence. Multi-run
  averaging (or a larger held-out set) would give a more reliable
  estimate of whether a prompt fix genuinely shifts behavior.
- **Test set size.** 18 questions is enough to catch clear failures but
  too small to produce statistically tight pass-rate estimates,
  particularly for the 5 outlier subtypes (roughly one question per
  subtype).
- **Retrieval fix not implemented.** q04's root cause (TF-IDF's lack of
  semantic matching) was diagnosed but not fixed in this task — the fix
  (swapping to a semantic embedding model) is a larger infrastructure
  change out of scope for a prompt-level checkpoint. It's documented as a
  recommendation in `failure_analysis.md`.

## 5. Files

- `test_set.json` — test set (Checkpoint 1)
- `scoring.py` — LLM-as-judge (Checkpoint 2)
- `run_eval.py`, `metrics.py` — eval runner and metrics
  (Checkpoint 3)
- `results.json`, `metrics.json` — raw results
- `failure_analysis.md` — root-cause analysis (Checkpoint 4)
- `prompt_v2.py` — improved prompt (Checkpoint 5)
- `compare_prompts.py`, `run_q06_demo.py` — before/after
  validation scripts
- `prompt_comparison_results.json`,
  `q06_demonstration.json` — before/after results
