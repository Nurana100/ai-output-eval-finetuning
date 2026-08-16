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

## Common Root Cause

Both failures trace back to retrieval, not generation:
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

No fine-tuning or prompt changes are indicated by these two failures — the
generation model performed correctly given its input in both cases.
