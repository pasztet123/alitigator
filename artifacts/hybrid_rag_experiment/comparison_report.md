# Hybrid Authority RAG Comparison

Status: not completed in the interactive run.

The A/B/C benchmark runner has been implemented at `apps/api/scripts/run_hybrid_authority_benchmark.py`, but a one-case dev smoke against `apps/api/data/processed/rag_eval_cases.sample.json` did not complete within the interactive verification window. The interrupted runs were inside local SQLite candidate generation. No holdout file was used as the cases source.

## What Exists

- Variant A: baseline retrieval using the current `search_chat_chunks` control path.
- Variant B: `hybrid_authority` retrieval without clarifier.
- Variant C: `hybrid_authority` retrieval with fixture clarifications when present, otherwise questions are traced and retrieval uses the raw query.
- Metrics schema for authority recall, primary-law recall, wrong-neighbor rate, issue coverage, authority type coverage, extraction placeholders, latency, and answer metrics placeholders.

## Result State

No aggregate case metrics are reported here because the real dev benchmark did not finish in this turn. The runner will overwrite `comparison_results.json` and this report when it completes.

## Recommended Run

From `apps/api`:

```bash
env PYTHONPATH=. \
  ALITIGATOR_RAG_CROSS_ENCODER_ENABLED=false \
  HYBRID_RAG_FAST_SQL_PRIMARY=true \
  HYBRID_RAG_FAST_SQL_AUTHORITY=true \
  .venv/bin/python scripts/run_hybrid_authority_benchmark.py \
    --cases data/processed/rag_eval_cases.sample.json \
    --max-cases 10 \
    --artifact-root ../../artifacts/hybrid_rag_experiment
```

For final experiment numbers, use the appropriate seed/dev case file and holdout only as `--exclude-cases`, never as `--cases`.
