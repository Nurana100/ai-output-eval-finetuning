"""Aggregate metrics from a list of per-question result records."""

# gemini-3.5-flash-lite standard paid-tier pricing (as of Aug 2026).
# This estimate shows what the run WOULD cost on the paid tier -- if you
# ran this on the free tier (as recommended in the README), your actual
# dollar cost was $0; this number exists so "average token cost" is a
# meaningful monitored metric regardless of which tier you're on.
# Check ai.google.dev/gemini-api/docs/pricing before trusting this for real budgeting.
PRICE_PER_MTOK_INPUT = 0.30
PRICE_PER_MTOK_OUTPUT = 2.50


def estimate_cost(input_tokens, output_tokens):
    return (input_tokens / 1_000_000) * PRICE_PER_MTOK_INPUT + (
        output_tokens / 1_000_000
    ) * PRICE_PER_MTOK_OUTPUT


def compute_metrics(records):
    """records: list of dicts each containing at least
    score (0/1/2 or None), latency_seconds, input_tokens, output_tokens,
    judge_input_tokens, judge_output_tokens.
    """
    n = len(records)
    scored = [r for r in records if r["score"] is not None]
    n_pass = sum(1 for r in scored if r["score"] == 2)
    n_partial = sum(1 for r in scored if r["score"] == 1)
    n_fail = sum(1 for r in scored if r["score"] == 0)

    total_latency = sum(r["latency_seconds"] for r in records)
    total_cost = sum(
        estimate_cost(
            r["input_tokens"] + r.get("judge_input_tokens", 0),
            r["output_tokens"] + r.get("judge_output_tokens", 0),
        )
        for r in records
    )

    return {
        "n_questions": n,
        "n_scored_by_judge": len(scored),
        "n_judge_parse_failures": n - len(scored),
        "pass_rate_strict": round(n_pass / n, 3) if n else None,          # score == 2 only
        "pass_rate_lenient": round((n_pass + n_partial) / n, 3) if n else None,  # score >= 1
        "n_pass": n_pass,
        "n_partial": n_partial,
        "n_fail": n_fail,
        "avg_latency_seconds": round(total_latency / n, 3) if n else None,
        "total_estimated_cost_usd": round(total_cost, 4),
        "avg_estimated_cost_usd_per_question": round(total_cost / n, 5) if n else None,
    }
