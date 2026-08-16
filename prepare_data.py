"""
Builds a small fine-tuning dataset targeting ONE failure category identified
in eval/failure_analysis.md: the base model sometimes answers confidently
on out-of-scope questions (hallucinating a feature/policy) instead of
declining. That's the failure this fine-tune corrects.

Training format: (question + retrieved context) -> target answer, where
out-of-scope examples are explicitly paired with a clear decline response,
and in-scope examples keep their normal grounded answers. This teaches the
model the *behavior* (decline when context doesn't cover it) rather than
just memorizing the 18-question test set — note the training examples
below are DIFFERENT questions from eval/test_set.json, precisely to avoid
test-set pollution (see report.md).
"""
import json
import os

from agent import retrieve

DECLINE_TEXT = "I don't have information about that in what I know about Nimbus Notes."

# In-scope training questions (NOT from eval/test_set.json)
IN_SCOPE = [
    ("What's included in the Team plan?", "Everything in Pro, plus shared workspaces, unlimited version history, and admin controls, for $15/user/month."),
    ("How do I reset my password?", "Use the 'Forgot password' link on the login screen. This does not apply if you signed up with Google or Apple sign-in."),
    ("Is two-factor authentication available?", "Yes, on all plans, via an authenticator app or SMS."),
    ("What happens if I upgrade my plan mid-cycle?", "The upgrade takes effect immediately with a prorated charge."),
    ("Does Nimbus Notes store EU user data in the EU?", "Yes, EU user data is stored in EU-based data centers for GDPR compliance."),
    ("What's the attachment size limit on the Free plan?", "5MB per note."),
]

# Out-of-scope training questions the model should learn to decline on
OUT_OF_SCOPE = [
    "Can Nimbus Notes translate my notes into other languages automatically?",
    "Does Nimbus Notes have a built-in calendar or reminders feature?",
    "What's the CEO of Nimbus Notes' name?",
    "Can I use Nimbus Notes to make video calls with my team?",
    "Does Nimbus Notes support voice-to-text dictation?",
    "What programming language is the Nimbus Notes app written in?",
]


def build():
    examples = []
    for q, a in IN_SCOPE:
        ctx = retrieve(q, k=2)
        context_text = "\n\n".join(c["text"] for c in ctx)
        examples.append({"question": q, "context": context_text, "answer": a})

    for q in OUT_OF_SCOPE:
        ctx = retrieve(q, k=2)  # retrieval will return something even if irrelevant -- realistic
        context_text = "\n\n".join(c["text"] for c in ctx)
        examples.append({"question": q, "context": context_text, "answer": DECLINE_TEXT})

    out_path = os.path.join(os.path.dirname(__file__), "train_data.jsonl")
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} training examples to {out_path}")


if __name__ == "__main__":
    build()
