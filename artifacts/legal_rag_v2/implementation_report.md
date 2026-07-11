# Legal RAG v2 implementation report

Date: 2026-07-11.

## Delivered

- Provider-neutral model gateway with per-stage model configuration, OpenAI
  Responses Structured Outputs, bounded retries and technical-only provider
  fallback.
- Model-first legal planner producing a fully typed `LegalResearchPlan` with
  source-grounded facts, issue decomposition, clarification and query families.
- Two retrieval lanes per issue, policy-free SQLite/MySQL FTS backend, RRF,
  optional versioned real-vector index and transparent rerank reasons.
- Provision-level chunk metadata and stable identities for article, paragraph,
  section, point and letter units in statutes and treaties.
- Model-based `AuthorityCard` extraction with exact span validation and an
  explicit heuristic fallback.
- Temporal provision graph, evidence bundles, structured claim synthesis,
  deterministic evidence/claim/writer/render validators and an answer plan.
- One v2 route before all legacy special routers, plus reversible `legacy`,
  `shadow` and `legal_rag_v2` modes.
- Atomic redacted traces, a resumable separate embedding-index command and a
  development-only A/B/C runner with a hard holdout path guard.
- User-visible application version advanced to `9.20.0`.

## Removed from the v2 decision path

The legacy document IDs, exact signature boosts, topic-specific query targets,
article nominations and four special-case early returns are not imported by or
called from v2. They remain available only in the legacy path and in the
explicitly marked planner fallback adapter. Calculators and deterministic
validators remain reusable, but no calculator is selected by benchmark phrase.

## Evaluation status

- Legacy pre-change baseline: 65 backend unit tests passed.
- Post-refactor backend suite: 113/113 tests pass; no protected holdout was
  touched.
- The offline reindex smoke test completed against a separate temporary index.
- Variant A ran on one public development case with local SQLite and the
  cross-encoder disabled: authority recall@5 and recall@20 were both `1.0`, six
  candidates were returned and latency was 65.210 s. This is a smoke result,
  not a representative quality claim. Full B/C quality, latency and cost
  measurements require a newly rotated provider key.
- Token usage and monetary cost are intentionally reported as unavailable
  until the gateway exposes provider usage metadata. No synthetic estimate is
  substituted.

## Remaining risks and switch recommendation

Do not switch production directly to v2 yet. The safe next state is `shadow`
after secret rotation and a real embedding reindex. Promotion should require
acceptable controlling-provision and authority recall, zero unsupported
material claims, source-span coverage review, acceptable latency, measured
provider cost and manual inspection of representative traces. The production
default remains `legacy` and rollback is immediate through one environment
flag.
