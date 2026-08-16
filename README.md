# Nimbus Notes RAG — Evaluation & Fine-tuning

A small RAG support agent (fictional product "Nimbus Notes") plus a full
evaluation framework: test set, LLM-as-judge scoring, metrics, failure
analysis, and a fine-tune that corrects one identified failure category.
Generation and judging use the Gemini API (free tier).

## What's in here

```
account_management.md, features.md, pricing.md,
privacy_security.md, troubleshooting.md   5 markdown docs the RAG agent retrieves from
build_index.py           builds a TF-IDF + FAISS retrieval index (offline, no downloads)
agent.py                 retrieval + Claude generation
test_set.json             18 Q&A pairs: 13 normal + 5 outliers
scoring.py                LLM-as-judge scoring (0/1/2 rubric)
metrics.py                pass rate / latency / cost aggregation
run_eval.py               orchestrator: runs the full eval, writes results.json + metrics.json
failure_analysis.md       3 documented failure cases with root cause
report.md                 written summary (fill in after running)
prepare_data.py           builds fine-tuning data for flan-t5-small
train.py                  fine-tunes flan-t5-small to fix one failure category (run in Colab)
compare_prompts.py        few-shot / prompt-version comparison
run_q06_demo.py           before/after demonstration for the q06 failure case
```

## Setup

```
pip install -r requirements.txt
cp .env.example .env
# edit .env and put your real GEMINI_API_KEY in there
```

Get a free key at <https://aistudio.google.com/apikey> if you don't have one
(no credit card required for the free tier).

**Do not paste your API key anywhere except your local `.env` file.** It's
in `.gitignore` so it won't get committed.

## Running the evaluation

```
python build_index.py     # builds the local retrieval index from the knowledge base docs
python run_eval.py        # runs all 18 questions through the agent + judge
```

This writes `results.json` (per-question detail) and `metrics.json` (aggregate pass rate / latency / cost). Sample output:

```
[q01] score=2  What does the Pro plan cost per month?
[q02] score=2  How long is the free trial for paid plans?
...
--- METRICS ---
{
  "n_questions": 18,
  "pass_rate_strict": 0.83,
  ...
}
```

After that, open `results.json`, find 2 more failure cases (score 0 or
1 — see the "suggested places to look" section at the bottom of `failure_analysis.md`), and fill them into `failure_analysis.md`.
Then fill in `report.md` with the real metrics numbers.

## Running the fine-tune (Checkpoint 5)

This needs to download a pretrained model from Hugging Face, so run it in **Google Colab** (free GPU, full internet) rather than locally unless you
already have a GPU set up:

1. `python prepare_data.py` (locally — this only uses the offline
retrieval index, no API key needed) → produces `train_data.jsonl`
2. Open colab.research.google.com, new notebook, Runtime > T4 GPU
3. `!pip install -q transformers datasets accelerate`
4. Upload `train_data.jsonl` to the Colab file panel
5. Paste in `train.py` and run it
6. It prints before/after answers on 2 probe questions and saves
`before_after_comparison.json` — bring that back into your report

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
  "pass_rate_strict": 0.78,
  "pass_rate_lenient": 0.94,
  "avg_latency_seconds": 1.42,
  "total_estimated_cost_usd": 0.0113
}
```

(Numbers above are illustrative of the *shape* of the output — your real
run will produce the actual numbers.)

## Design notes / why things are built this way

- **TF-IDF + FAISS instead of neural embeddings:** keeps the whole
retrieval pipeline runnable offline with zero model downloads, which
also makes it fast to iterate on. Documented tradeoff: it's weak on
paraphrase (see Failure Case 1) — that's flagged, not hidden.
- **LLM-as-judge bias:** documented explicitly in `scoring.py` and
`report.md` — judge scores are not treated as ground truth without
a manual spot-check.
- **Test-set pollution avoided:** the fine-tuning examples in
`prepare_data.py` are different questions from
`test_set.json`, so the before/after comparison isn't validated on
data the fine-tune already saw.
