# Failure Analysis — Checkpoint 4

## Summary

Two real failure cases were pulled from the full 18-question evaluation run
(`eval/results.json`). Both point to the same underlying issue: **retrieval
coverage/chunking**, not the generation model reasoning incorrectly about
what it was given.

| ID  | Question | Score | Failure Type |
|-----|----------|-------|---------------|
| q04 | Can I use Nimbus Notes without an internet connection? | 0 | Retrieval — relevant chunk not surfaced |
| q06 | How do I export my notes and what formats are supported? | 1 | Retrieval — partial coverage, key qualifier missing |
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
- **Retrieved sources:** `troubleshooting.md`, `features.md`,
  `troubleshooting.md` (duplicate).
- **Root cause:** `features.md` was retrieved, but the specific chunk
  containing the offline-mode fact evidently wasn't part of what was
  returned to the model — otherwise the model would have used it. The
  duplicate `troubleshooting.md` entry wasted a retrieval slot that could
  have surfaced a second, more relevant `features.md` chunk instead.
- **Classification:** Retrieval/chunking failure, not a generation failure.
  The model correctly declined to hallucinate an answer it didn't have —
  the problem is what it was given, not what it did with it.

## q06 — "How do I export my notes and what formats are supported?"

- **Expected answer:** Settings > Export; supports Markdown, plain text, or
  PDF; bulk export of the whole account as a .zip is Pro/Team only.
- **Model answer:** Correctly described the navigation path and all three
  export formats, but omitted the Pro/Team-only bulk export detail.
- **Retrieved sources:** `troubleshooting.md`, `features.md`,
  `privacy_security.md`.
- **Root cause:** The core export info was retrieved and used correctly.
  The missing qualifier (bulk export tier restriction) likely lives in a
  different chunk of `features.md` that didn't make the top-k cut —
  `privacy_security.md` was retrieved instead of that second chunk, and
  wasn't relevant to the question.
- **Classification:** Retrieval coverage failure. The model generated
  faithfully from incomplete context.

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

## Common Root Cause

q04 and q06 trace back to retrieval, not generation:
- Chunking appears to be splitting single features (offline mode
  availability, export tier restrictions) away from their qualifying
  details, so a chunk can contain the "what" without the "who/which plan."
- Top-k retrieval count and/or duplicate-chunk filtering may be too
  permissive, letting a duplicate or lower-relevance chunk take a slot
  that a more relevant chunk should have filled.

## Recommended Fixes

1. **Increase top-k** for retrieval so more candidate chunks are available
   per query, reducing the chance the right chunk is cut off.
2. **De-duplicate retrieved chunks** before passing them to the generator
   (q04 retrieved the same `troubleshooting.md` chunk twice).
3. **Review chunk size/overlap** for `features.md` — features and their
   plan-tier qualifiers should ideally live in the same chunk, or overlap
   should be large enough to keep them together.
4. **Re-run q04 and q06 after any retrieval change** to confirm the fix
   actually surfaces the missing information, rather than assuming it will.

For q17's judge-leniency issue specifically (a separate root cause from
q04/q06, so it needs a separate fix — not solved by retrieval changes):

5. **Tighten the judge rubric for trap/discriminative questions.** Add an
   explicit instruction (or a separate rubric field) requiring the judge to
   check whether the answer addresses any distractor/discriminating detail
   named in the reference answer, not just whether the final factual claim
   matches.
6. **Track trap questions as their own metric slice** rather than folding
   them into the overall pass rate, so judge leniency on this question type
   doesn't get averaged away by the rest of the (largely non-trap) test set.

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

No fine-tuning or prompt changes to the *generation* model are indicated
by q04/q06 — the model performed correctly given its input in both cases.
Checkpoint 5 will address the retrieval fixes above using a held-out
question set, per the task's guidance against validating a fix on the
same samples used to make it.
