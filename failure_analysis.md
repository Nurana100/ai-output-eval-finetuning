# Failure Analysis — Checkpoint 4

## Summary

Three real cases were identified from the full 18-question evaluation run
(`eval/results.json`), spanning three distinct root causes: retrieval
(q04), generation/prompt (q06), and judge-scoring bias (q17). An earlier
draft of this analysis mis-classified q06 as a retrieval failure; a deeper
diagnostic (printing full retrieved-chunk text, not just source filenames)
showed the correct chunk **was** retrieved with the full answer inside it
— the model simply left a detail out despite having it. That distinction
matters because it points to a different fix (prompting vs. retrieval).

| ID  | Question | Score | Failure Type |
|-----|----------|-------|---------------|
| q04 | Can I use Nimbus Notes without an internet connection? | 0 | Retrieval — relevant chunk never surfaced (lexical/TF-IDF vocabulary mismatch) |
| q06 | How do I export my notes and what formats are supported? | 1 | Generation — model had the full answer in context and still omitted a detail |
| q17 | If I cancel my monthly plan on day 10 of a 30-day cycle, do I get a partial refund for the remaining 20 days? | 2 (judge) | LLM-judge leniency on a trap question — scored correct despite missing the discriminating detail |

**Note:** q17 was not flagged by the automated scorer. It surfaced during a
manual review of outlier/trap questions specifically because the task's
Quality Checks call out LLM-as-judge bias as a known limitation to watch
for — this case is a concrete example of that bias, not a retrieval or
generation bug like q04/q06.

## q04 — "Can I use Nimbus Notes without an internet connection?"

- **Expected answer:** Yes, offline mode is available on the Pro and Team
  plans; edits sync automatically once back online.
- **Model answer:** Said the information wasn't in the provided context,
  citing only a troubleshooting note about checking your internet
  connection.
- **Retrieved sources (k=3):** `troubleshooting.md`, `features.md`,
  `troubleshooting.md` (duplicate — the duplicate is itself a minor bug;
  see Recommended Fixes).
- **Diagnostic:** Re-ran retrieval at k=10 and printed full chunk text
  (not just filenames). The chunk that actually contains the offline-mode
  answer — `"Offline mode: Available on Pro and Team plans. Notes edited
  offline sync automatically once you're back online."` — does **not**
  appear anywhere in the top 10 results for this query; it is the lowest-
  scoring chunk in the whole corpus for it (score 0.0132, last place).
  Meanwhile a near-empty chunk containing only the document header
  (`"# Nimbus Notes — Features"`) ranks 2nd (score 0.2159), and
  `troubleshooting.md`'s syncing section ranks 1st (0.2888) purely because
  it shares the literal words "internet" and "connection" with the query.
- **Root cause:** The retriever is TF-IDF (pure lexical/keyword overlap,
  per `build_index.py`), which has no notion of synonymy. The question
  says "without an internet connection"; the answer is filed under
  "offline mode" — different vocabulary, same concept. TF-IDF has no way
  to connect the two, so a document that merely shares surface words
  (the sync-troubleshooting section) outranks the chunk with the actual
  answer. This is compounded by a chunking artifact: `chunk_text()`
  sometimes emits a chunk containing only a markdown header, and that
  near-empty chunk still ranks highly because TF-IDF/cosine similarity
  favors short documents when even a couple of terms match.
- **Classification:** Retrieval failure — specifically, a lexical-search
  vocabulary mismatch, not a chunking-boundary problem. Increasing top-k
  alone would not fix this reliably, since the correct chunk isn't merely
  low-ranked, it's below the entire top-10 in this run.

## q06 — "How do I export my notes and what formats are supported?"

- **Expected answer:** Settings > Export; supports Markdown, plain text, or
  PDF; bulk export of the whole account as a .zip is Pro/Team only.
- **Model answer:** Correctly described the navigation path and all three
  export formats, but omitted the Pro/Team-only bulk export detail.
- **Retrieved sources (k=3):** `troubleshooting.md`, `features.md`,
  `privacy_security.md`.
- **Diagnostic:** Re-ran retrieval and printed full chunk text. The
  top-ranked chunk (`troubleshooting.md`, score 0.4699 — the highest score
  of any chunk for this query) contains the **complete** answer, including
  the exact missing detail: `"Bulk export of the entire account as a .zip
  is available on Pro and Team plans only."` This chunk was retrieved and
  was in the model's context window.
- **Root cause:** This is not a retrieval failure — correcting an earlier
  draft of this analysis, which mis-attributed it to missing context. The
  model had the full answer available and still dropped the tier
  qualifier when generating its (intentionally concise, per the system
  prompt's "2-4 sentences" instruction) response. This looks like a
  generation/summarization failure under a length constraint: the model
  prioritized the parts of the answer that directly matched the question's
  literal wording ("how," "what formats") over a secondary qualifier not
  explicitly asked about.
- **Classification:** Generation/prompt failure, not retrieval. The fix
  belongs in the prompt (e.g., an explicit instruction or few-shot example
  showing that plan-tier restrictions must be included when present),
  not in the retrieval pipeline.

## q17 — "If I cancel my monthly plan on day 10 of a 30-day cycle, do I get a partial refund for the remaining 20 days?" (judge-scoring limitation)

- **Type:** `outlier_trap` — this question was deliberately written to test
  whether the model conflates two similar-sounding policies: (a) no partial
  refund on mid-cycle cancellation, and (b) the 14-day money-back guarantee
  on *new* subscriptions.
- **Expected answer:** States there's no partial refund, the plan stays
  active until the end of the billing period, and explicitly distinguishes
  this from the unrelated 14-day money-back guarantee.
- **Model answer:** "No, you do not get a partial refund for the unused
  portion of a monthly plan. Your plan will simply remain active until the
  end of the current billing period." — factually correct, but never
  mentions the 14-day guarantee or shows it recognized the trap at all.
- **Judge score:** 2/2. **Judge justification:** the answer "correctly
  states that there is no partial refund and the plan remains active until
  the end of the billing period."
- **Root cause:** This is not a retrieval or generation bug — the model's
  answer is factually accurate. It's a **judge-scoring limitation**: the
  automated LLM-as-judge evaluates surface correctness against the
  reference answer's factual claims, but doesn't penalize the omission of
  the reasoning step that would prove the model actually resolved the
  trap (versus just not tripping it by chance). A model that silently
  ignores the distractor policy and a model that actively reasons through
  it and rejects it look identical to this judge.
- **Classification:** LLM-as-judge bias/leniency, specifically on trap
  questions designed to test discriminative reasoning rather than plain
  factual recall. This matches the known judge-bias limitation called out
  in the task's Quality Checks (judges tend to reward answers that hit the
  key facts, independent of whether the answer demonstrates the reasoning
  those facts were meant to test).
- **Implication:** Aggregate pass-rate metrics (Checkpoint 3) may be
  slightly optimistic on trap/discriminative questions specifically,
  because the judge rubric doesn't check for demonstrated disambiguation,
  only final-answer correctness.

## Root Causes (Three Distinct Failure Modes)

Diagnosis confirmed these are three separate problems requiring three
separate fixes — not one shared cause:

- **q04 — retrieval (lexical mismatch):** TF-IDF retrieval matches surface
  vocabulary, not meaning. "Without an internet connection" and "offline
  mode" don't share words, so the correct chunk loses to an irrelevant
  chunk that happens to share literal terms with the question.
- **q06 — generation (context underuse):** The correct chunk was
  retrieved and available; the model still dropped a qualifying detail,
  likely due to the system prompt's brevity constraint competing with
  completeness.
- **q17 — judge-scoring bias:** Covered above; a rubric problem, not a
  retrieval or generation problem.

## Recommended Fixes

**For q04 (retrieval):**
1. Swap TF-IDF for a semantic embedding model (e.g. sentence-transformers),
   which `build_index.py`'s own docstring already anticipates as a drop-in
   replacement — `retrieve()`'s interface wouldn't need to change.
2. Short-term, cheaper mitigation without a new model: filter out
   near-empty chunks (e.g. below a minimum token/character count) so
   header-only chunks like `"# Nimbus Notes — Features"` can't outrank
   substantive content.
3. De-duplicate retrieved chunks before passing them to the generator
   (q04 retrieved the same `troubleshooting.md` chunk twice, wasting a
   context slot).

**For q06 (generation):**
4. Add a few-shot example or explicit instruction to the system prompt
   telling the model to include plan-tier/eligibility qualifiers whenever
   the retrieved context contains them, even under the brevity constraint.
   (This is the fix implemented for Checkpoint 5 — see
   `finetune/` or prompt-comparison artifacts for before/after results on
   a held-out question set.)

**For q17 (judge bias):**
5. Tighten the judge rubric for trap/discriminative questions: require the
   judge to check whether the answer addresses any distractor/
   discriminating detail named in the reference answer, not just whether
   the final factual claim matches.
6. Track trap questions as their own metric slice rather than folding them
   into the overall pass rate, so judge leniency on this question type
   doesn't get averaged away by the rest of the (largely non-trap) set.

## Note on LLM-as-Judge Limitations

As called out in this task's Quality Checks, an LLM-as-judge has known
biases (e.g., leniency, preference for longer or more confident-sounding
answers). q17 is a concrete instance of this: the judge scored a factually
correct but reasoning-incomplete answer as fully correct (2/2) because its
rubric only checks final-answer correctness, not whether the answer
demonstrates it resolved the specific ambiguity the question was designed
to test. This means the aggregate metrics in Checkpoint 3 should be read
as an upper bound on trap-question performance, not a guarantee that the
model is reliably distinguishing similar policies — manual spot-checking
of outlier/trap cases remains necessary and was how q17 was caught here.

## Note on Generation Non-Determinism (Discovered During Checkpoint 5)

While validating the q06 fix, re-running the *original, unmodified*
baseline prompt against q06 produced a **different score than the
original eval run**: score 2 (full credit, qualifier included) on this
later run, versus score 1 (qualifier dropped) in the run originally
documented above and in `eval/results.json`. Same prompt, same question,
same retrieved context — different output. This is expected LLM sampling
variance, not a bug, but it has a real implication: **a single
before/after comparison on one sample is not reliable evidence that a
prompt fix works or doesn't**, since the baseline itself doesn't
reproduce identically run to run.

This is why Checkpoint 5's validation leans on a held-out multi-question
set rather than a single q06 before/after: an aggregate across several
questions is far less sensitive to one sample's random variance than a
single data point is. It also means the true baseline failure rate for
q06-type omissions is probably better estimated by running the question
multiple times than by a single pass — a good candidate for future work
if more API quota is available.

## Checkpoint 5 Results

**Held-out validation set** (q09, q10, q12, q16 — NOT used to design the
fix): baseline avg score 2.00, improved avg score 2.00 across all four.
No regressions from the fix. All four already scored full credit before
the change, so this run doesn't show the fix creating new failures on
questions with a similar qualifier-dropping risk shape.

**q06 direct comparison** (the diagnosed sample — reported separately
from the held-out set to avoid treating it as validation proof, per the
anti-pollution guidance): baseline scored 2 and improved scored 2 on this
run, both including the Pro/Team qualifier. Given the non-determinism
noted above, this single run does not confirm the fix causes the
improvement (the original documented failure showed the *unmodified*
prompt already capable of scoring 1 on this exact question). What the fix
does provide is an explicit instruction plus a worked example telling the
model to preserve qualifiers under the brevity constraint, converting
qualifier-inclusion from incidental (works or doesn't depending on
sampling) to instructed behavior — the mechanism is sound even though a
single-sample before/after can't statistically prove it here.

Checkpoint 5 implements the few-shot prompt fix for q06's failure category
(generation/context-underuse), validated on a held-out question set that
was not used while designing the fix, per the task's guidance against
validating a fix on the same samples used to make it. q04's retrieval fix
(swapping TF-IDF for semantic embeddings) is documented here as a
recommendation but is a larger infrastructure change out of scope for a
prompt-level Checkpoint 5 fix.

**Update per mentor review (8/17/2026):** the original held-out run (4
questions, 1 trial each) produced baseline == improved on every question,
which only demonstrates "no regression," not "measurable improvement,"
and — per the non-determinism note above — a single sample per question
isn't strong evidence either way. `compare_prompts.py` was revised to (a)
expand the held-out set from 4 to 7 qualifier-bearing questions and (b)
run each question 3 times per prompt version and average, so the
before/after conclusion is less sensitive to one sample's random
variance. Re-run `python compare_prompts.py` with a live `GEMINI_API_KEY`
to regenerate `prompt_comparison_results.json` and update the numbers in
`report.md` accordingly — the numbers above reflect the pre-revision run.
