# Nimbus Notes RAG — Evaluation & Fine-tuning

A small RAG support agent (fictional product "Nimbus Notes") plus a full
evaluation framework: test set, LLM-as-judge scoring, metrics, failure
analysis, and a fine-tune that corrects one identified failure category.
Generation and judging use the Gemini API (free tier).

## What's in here

```
data/knowledge_base/   5 markdown docs the RAG agent retrieves from
rag/build_index.py     builds a TF-IDF + FAISS retrieval index (offline, no downloads)
rag/agent.py            retrieval + Claude generation
eval/test_set.json      18 Q&A pairs: 13 normal + 5 outliers
eval/scoring.py          LLM-as-judge scoring (0/1/2 rubric)
eval/metrics.py          pass rate / latency / cost aggregation
eval/run_eval.py         orchestrator: runs the full eval, writes results.json + metrics.json
eval/failure_analysis.md 3 documented failure cases with root cause
eval/report.md           written summary (fill in after running)
finetune/                fine-tunes flan-t5-small to fix one failure category (run in Colab)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env and put your real GEMINI_API_KEY in there
```

Get a free key at https://aistudio.google.com/apikey if you don't have one
(no credit card required for the free tier).

**Do not paste your API key anywhere except your local `.env` file.** It's
in `.gitignore` so it won't get committed.

## Running the evaluation

```bash
python rag/build_index.py     # builds data/index/ from the knowledge base docs
python eval/run_eval.py       # runs all 18 questions through the agent + judge
```

This writes `eval/results.json` (per-question detail) and
`eval/metrics.json` (aggregate pass rate / latency / cost). Sample output:

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

After that, open `eval/results.json`, find 2 more failure cases (score 0 or
1 — see the "suggested places to look" section at the bottom of
`eval/failure_analysis.md`), and fill them into `eval/failure_analysis.md`.
Then fill in `eval/report.md` with the real metrics numbers.

## Running the fine-tune (Checkpoint 5)

This needs to download a pretrained model from Hugging Face, so run it in
**Google Colab** (free GPU, full internet) rather than locally unless you
already have a GPU set up:

1. `python finetune/prepare_data.py` (locally — this only uses the offline
   retrieval index, no API key needed) → produces `finetune/train_data.jsonl`
2. Open colab.research.google.com, new notebook, Runtime > T4 GPU
3. `!pip install -q transformers datasets accelerate`
4. Upload `train_data.jsonl` to the Colab file panel
5. Paste in `finetune/train.py` and run it
6. It prints before/after answers on 2 probe questions and saves
   `before_after_comparison.json` — bring that back into your report

## Sample request/response log

```
$ python eval/run_eval.py
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
- **LLM-as-judge bias:** documented explicitly in `eval/scoring.py` and
  `eval/report.md` — judge scores are not treated as ground truth without
  a manual spot-check.
- **Test-set pollution avoided:** the fine-tuning examples in
  `finetune/prepare_data.py` are different questions from
  `eval/test_set.json`, so the before/after comparison isn't validated on
  data the fine-tune already saw.
