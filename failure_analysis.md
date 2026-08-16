# Failure Case Analysis

Instructions: run `python eval/run_eval.py` first (needs `ANTHROPIC_API_KEY` set),
then open `eval/results.json` and fill in cases 2 and 3 below with real examples
from your run — pick the ones with `score: 0` or `score: 1`. Case 1 is already
filled in from a *retrieval-level* check I ran directly against the index
(no API key needed), so it's real and reproducible — try it yourself with:

```
python -c "from rag.agent import retrieve; [print(r['source'], round(r['score'],3)) for r in retrieve('how much for pro tho', k=5)]"
```

---

## Case 1: Retrieval — vocabulary mismatch on informal phrasing (q14)

**Question:** "how much for pro tho"
**Expected:** Should retrieve `pricing.md` (Pro is $8/month).

**What happened:** TF-IDF retrieval ranked `pricing.md` well below `account_management.md`
and `features.md` for this phrasing. For the more formal version of the same question
("Nimbus Notes Pro pricing per month"), `pricing.md` ranks #1 by a wide margin
(0.44 vs 0.09 similarity score). The informal version shares almost no
distinctive vocabulary with the pricing doc's dollar-amount-heavy text, so
TF-IDF — which matches surface word overlap, not meaning — fails to connect them.

**Root cause:** Poor retrieval, specifically: **TF-IDF has no notion of
synonymy or informal/formal paraphrase**, only exact (stemmed) token overlap.
This is an inherent limitation of the retrieval method chosen for this
project (chosen for offline reproducibility — see README), not a prompt or
generation issue downstream.

**Category:** `retrieval`

---

## Case 2: [fill in from your eval/results.json]

**Question:**
**Expected:**
**Model answer:**
**Judge score:**

**What happened:**

**Root cause (poor retrieval / poor prompt / unclear question / other):**

**Category:**

---

## Case 3: [fill in from your eval/results.json]

**Question:**
**Expected:**
**Model answer:**
**Judge score:**

**What happened:**

**Root cause (poor retrieval / poor prompt / unclear question / other):**

**Category:**

---

## Suggested places to look for cases 2 and 3

Based on how the test set was designed, these `type` values in test_set.json
are the most likely to surface real failures — check those rows in
`results.json` first:

- `outlier_out_of_scope` (q15, q18) — does the model correctly decline, or
  does it hallucinate a plausible-sounding but made-up answer?
- `outlier_trap` (q17) — does the model conflate the 14-day money-back
  guarantee with the "no partial refund on cancellation" policy? These are
  two different, easily-confused policies in the same knowledge base.
- `outlier_ambiguous` (q16) — does the model's answer actually resolve the
  ambiguity in the question, or does it dodge part of it?
