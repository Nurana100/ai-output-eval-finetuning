# Nimbus Notes RAG — Evaluation & Fine-tuning

A small RAG support agent (fictional product "Nimbus Notes") plus a full
evaluation framework: test set, LLM-as-judge scoring, metrics, failure
analysis, and a targeted fix for one identified failure category with
held-out before/after validation. Generation and judging both use the
Gemini API (free tier).

## What's in here

All files sit at the repo root (no subfolders):

```
account_management.md, features.md, pricing.md,      knowledge base — 5 docs
  privacy_security.md, troubleshooting.md             the RAG agent retrieves from

build_index.py          builds a TF-IDF + FAISS retrieval index into index/ (offline, no downloads)
agent.py                retrieval + Gemini generation
test_set.json            18 Q&A pairs: 13 normal + 5 outliers                    (Checkpoint 1)
scoring.py               LLM-as-judge scoring (0/1/2 rubric)                     (Checkpoint 2)
run_eval.py              orchestrator: runs the full eval, writes
                          results.json + metrics.json                           (Checkpoint 3)
metrics.py               pass rate / latency / cost aggregation
failure_analysis.md      3 documented failure cases with root cause              (Checkpoint 4)
prompt_v2.py             improved system prompt, fixes the q06 failure category  (Checkpoint 5)
compare_prompts.py       held-out before/after validation (q09, q10, q12, q16)
run_q06_demo.py          separate before/after run on q06 itself (the diagnosed
                          sample, reported separately from the held-out set)
report.md                full written summary tying every checkpoint together   (Checkpoint 6)

prepare_data.py, train.py, train_data.jsonl   an earlier fine-tuning approach
                                               (flan-t5-small in Colab) that was
                                               superseded by the prompt_v2.py fix
                                               above -- kept for reference, not
                                               part of the submitted Checkpoint 5
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and put your real GEMINI_API_KEY in there
```

Get a free key at https://aistudio.google.com/apikey if you don't have one
(no credit card required for the free tier).

**Do not paste your API key anywhere except your local `.env` file.** Make
sure `.env` is listed in `.gitignore` before you commit, so it never gets
pushed.

## Running the evaluation

```bash
python build_index.py     # builds index/ from the 5 knowledge-base docs
python run_eval.py        # runs all 18 questions through the agent + judge
```

This writes `results.json` (per-question detail) and `metrics.json`
(aggregate pass rate / latency / cost). Sample output:

```
[q01] score=2  What does the Pro plan cost per month?
[q02] score=2  How long is the free trial for paid plans?
...
[q04] score=0  Can I use Nimbus Notes without an internet connection?
[q06] score=1  How do I export my notes and what formats are supported?
...
--- METRICS ---
{
  "n_questions": 18,
  "n_scored_by_judge": 18,
  "pass_rate_strict": 0.889,
  "pass_rate_lenient": 0.944,
  "avg_latency_seconds": 0.983,
  "total_estimated_cost_usd": 0.0065
}
```

(Exact numbers vary slightly run to run — Gemini generation is not fully
deterministic. See "Known Limitations" in `report.md`.)

## Reproducing the Checkpoint 5 fix

`prompt_v2.py` adds a few-shot instruction fixing the q06 failure category
(the model dropping a plan-tier/limit qualifier from an otherwise-correct
answer). To reproduce the held-out validation:

```bash
python compare_prompts.py   # baseline vs. improved prompt on q09, q10, q12, q16
                             # (held out -- none of these were used to design the fix)
python run_q06_demo.py      # baseline vs. improved prompt on q06 itself, reported
                             # separately since q06 is the diagnosed sample, not
                             # a validation sample
```

These write `prompt_comparison_results.json` and `q06_demonstration.json`.

## Sample request/response log

```
$ python run_eval.py
[q01] score=2  What does the Pro plan cost per month?
[q14] score=1  how much for pro tho
[q15] score=2  What's the weather like today?
...
--- METRICS ---
{
  "n_questions": 18,
  "n_scored_by_judge": 18,
  "pass_rate_strict": 0.889,
  "pass_rate_lenient": 0.944,
  "avg_latency_seconds": 0.983,
  "total_estimated_cost_usd": 0.0065
}
```

## Design notes / why things are built this way

- **TF-IDF + FAISS instead of neural embeddings:** keeps the whole
  retrieval pipeline runnable offline with zero model downloads, which
  also makes it fast to iterate on. Documented tradeoff: it's weak on
  paraphrase — see the q04 failure case in `failure_analysis.md`, where
  the query's wording shares no vocabulary with the answer's wording.
- **LLM-as-judge bias:** documented explicitly in `scoring.py` and
  `report.md` — judge scores are treated as an upper bound, not ground
  truth, and one case (q17) shows the judge giving full credit to an
  answer that skipped the actual reasoning it was meant to test.
- **Held-out validation for the fix:** `compare_prompts.py` scores the
  Checkpoint 5 fix only on questions that were *not* used to diagnose or
  design it, to avoid the cyclical-validation trap of "fixing on the test
  case, then testing on the same case."

## Full write-up

See `report.md` for the complete methodology, results, and known
limitations across all 6 checkpoints, and `failure_analysis.md` for the
full root-cause detail behind each of the three documented failures.
