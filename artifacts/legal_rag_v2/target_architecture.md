# Alitigator: target Legal RAG v2 architecture

Implementation date: 2026-07-11. The v2 architecture is additive and selected
only by `LEGAL_PIPELINE_MODE`; the legacy path remains intact for rollback.

## One request flow

```text
POST /api/chat
  -> auth, persistence, effective user question
  -> LEGAL_PIPELINE_MODE
       legacy: existing routers and writer
       shadow: run v2 asynchronously, return legacy response
       legal_rag_v2:
         LegalQueryPlanner (Structured Output)
         -> LegalResearchPlan + grounded fact spans
         -> primary-law lane per issue
         -> authority lane per issue
         -> lexical/vector fusion + transparent rerank
         -> model-extracted AuthorityCards + exact document spans
         -> temporal ProvisionGraph
         -> EvidenceBundle per issue
         -> deterministic calculation engine
         -> structured LegalClaims
         -> deterministic claim validation
         -> AnswerPlan
         -> structured answer writer
         -> deterministic renderer and post-render validation
         -> persisted stage trace
```

The v2 branch returns before every legacy special-case router. No bad-debt,
mixed-invoice, housing-relief or benchmark phrase can bypass the common v2
flow. Domain pipelines remain regression oracles and legacy rollback code.

## Boundaries

- `model_gateway.py` is the only provider boundary. OpenAI uses the Responses
  API and Pydantic Structured Outputs. Anthropic is an optional technical
  fallback, never a benchmark-result retry.
- `legal_rag_v2/planner.py` owns research intent, facts, issues, clarification
  and query families. Its legacy fallback is lazy, explicit and traced.
- `legal_rag_v2/backends.py` performs policy-free corpus FTS. It contains no
  benchmark query targets, document IDs or topic-specific article lists.
- `legal_rag_v2/retrieval.py` keeps primary law and authorities in independent
  lanes and combines lexical and real-vector ranks with RRF.
- `legal_rag_v2/authority.py` extracts typed authority positions and validates
  exact spans against the retrieved text.
- `legal_rag_v2/provision_graph.py` represents provisions and relationships at
  article/paragraph/section/point/letter granularity with effective periods.
- `legal_rag_v2/pipeline.py` is the only v2 orchestrator. Claims, citations and
  calculations must pass deterministic gates before rendering.
- `legal_rag_v2/trace.py` atomically writes redacted, parseable artifacts for
  every stage.

## Fallback policy

Allowed planner fallback reasons are provider timeout/unavailability, invalid
schema, low planner confidence, insufficient candidate presence and an
explicit forced test variant. Fallback may add query hints and candidates; it
cannot set the final legal conclusion. Provider fallback is allowed only for
technical errors. Schema and request errors fail closed after their own bounded
retry budgets.

Offline hash embeddings are never automatic. They require both
`--offline-hash` and `--allow-offline-hash`, or the corresponding explicit v2
development flag. Production indexing uses `text-embedding-3-large`.

## Temporal and evidence invariants

- A provision outside the plan's `target_date` remains visible in the graph as
  historical but cannot enter `controlling_provisions`.
- Primary law controls the normative answer. Interpretations and judgments are
  evidence of practice/reasoning, not substitutes for legislation.
- Every material approved/conditional claim binds to retrieved provision IDs
  and exact source spans. Authority-pattern claims additionally bind to real
  authority document IDs.
- Numeric conclusions require a deterministic `CalculationRecord`; the default
  calculation engine returns no result rather than asking a model to calculate.
- The writer receives validated claims and may not invent facts, sources,
  citations, calculations or new legal conclusions.

## Deployment sequence

1. Rotate all exposed provider secrets and configure the new key outside git.
2. Build the separate versioned embedding index; review its indexing report.
3. Run development A/B/C and inspect candidate recall before reranking.
4. Enable `shadow`, inspect traces and operational latency/cost telemetry.
5. Switch to `legal_rag_v2` only after quality gates pass; rollback is the
   single flag change back to `legacy`.

The protected holdout is outside this implementation workflow and was not
opened, modified or executed.
