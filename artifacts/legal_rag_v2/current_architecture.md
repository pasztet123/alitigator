# Alitigator: current legal RAG architecture

Audit date: 2026-07-11. This document describes the code before the
`legal_rag_v2` refactor. It intentionally records the baseline rather than a
target design.

## Runtime configuration observed locally

- `LEGAL_PIPELINE_MODE` is not implemented and is unset.
- `LEGAL_RETRIEVAL_MODE` is unset, therefore `baseline` is active.
- `ENABLE_LEGAL_CLARIFIER` is unset, therefore the hybrid clarifier is off.
- `ALITIGATOR_RAG_BACKEND=mysql` is active. MySQL credentials are configured;
  their values were not read into this document.
- Supabase account/storage credentials are configured, but
  `ALITIGATOR_RAG_USE_SUPABASE=false` and
  `ALITIGATOR_RAG_SUPABASE_SYNC=false`.
- Anthropic is configured and OpenAI is not. The default model comes from
  `ANTHROPIC_MODEL`, falling back to `claude-sonnet-4-6`.
- Existing baseline before edits: 65 `unittest` tests pass.

## Current request flow

```text
POST /api/chat
  -> authenticate, ensure profile, redact input
  -> create/load chat and persist pending user messages
  -> append UI intent hints to an effective retrieval query
  |
  +-> bad-debt benchmark trace detector
  |    -> controlled bad-debt pipeline -> deterministic renderer -> RETURN
  |
  +-> complete bad-debt detector
  |    -> controlled bad-debt pipeline -> claims/calculations/renderer -> RETURN
  |
  +-> mixed-invoice detector
  |    -> controlled mixed-invoice pipeline -> claims/renderer -> RETURN
  |
  +-> housing-relief detector
  |    -> controlled housing pipeline -> claims/calculations/renderer -> RETURN
  |
  -> retrieval
       +-> LEGAL_RETRIEVAL_MODE=hybrid_authority
       |    -> heuristic intent -> heuristic FactGraph/IssueGraph
       |    -> PrimaryLane and AuthorityLane
       |    -> heuristic AuthorityCard -> heuristic reranking/EvidenceBundle
       |
       +-> default baseline
            -> search_chat_chunks
            -> deterministic statute routing + typed statute/interpretation/
               judgment retrieval + handcrafted reranking
  -> primary-source fallback injection (including topic-specific bundles)
  -> KSeF-specific second retrieval when a hardcoded current bundle is absent
  -> optional Supabase fallback, but only when the configured backend is SQLite
  -> static LegalSourcePlan, axis coverage, legal-rule extraction, missing facts
  -> build_analysis_trace (claims are diagnostic, not a writer input gate)
  |
  +-> no Anthropic key: deterministic demo renderer -> validation -> RETURN
  |
  -> build a large case-specific system prompt and source context
  -> direct Anthropic Messages HTTP request from main.py
  -> free-form answer -> regex/deterministic guardrails -> render validation
  -> append retrieval citations -> charge credit -> persist -> RETURN
```

The prompt-hints endpoint is a separate path. It calls Anthropic directly from
`main.py`; on any error it returns regex/static questions.

## Early returns and bypasses

All line references below refer to the pre-refactor `apps/api/app/main.py`.

| Location | Trigger | Bypasses |
|---|---|---|
| 2815-2847 | special bad-debt benchmark trace phrase | normal retrieval, general writer, normal claim trace |
| 2849-2950 | `can_run_bad_debt_pipeline` | all general retrieval and the model writer |
| 2952-3012 | `is_mixed_invoice_query` | all general retrieval and the model writer |
| 3014-3098 | `can_run_housing_relief_pipeline` | all general retrieval and the model writer |
| 3236-3292 | no Anthropic API key | live model writer; returns the demo response after retrieval |
| 3493 | normal terminal response | final return after provider generation and persistence |

The first four are production routing decisions with no feature flag. They are
the special-pipeline early returns that must not exist in the v2 path.

## Where each stage currently happens

- Retrieval starts only after the special pipelines, around
  `main.py:3120-3181`. Baseline calls `search_chat_chunks`; the experimental
  branch calls `run_hybrid_authority_retrieval`.
- General legal rules and the diagnostic claim trace are built around
  `main.py:3183-3235` and `build_analysis_trace` at `main.py:991-1062`.
- Controlled claims are built inside `bad_debt_pipeline.py`,
  `housing_relief_pipeline.py`, and `controlled_legal_pipeline.py`.
- The live answer model is called around `main.py:3394-3411`. Prompt hints use
  a separate direct provider call around `main.py:805-905`.
- Controlled render validation lives in `controlled_legal_pipeline.py` and the
  controlled domain pipelines. The generic answer is checked by
  `enforce_reply_guardrails` and `validate_final_output` in `main.py`.
- Generic claims do not gate answer generation. The writer receives rules,
  retrieval context, missing-fact messages and hybrid evidence, but not a set
  of validated `LegalClaim` objects that it is forbidden to exceed.

## Retrieval backends

1. SQLite FTS is the development default in `.env.example`; `rag.py` owns its
   schema, reindexing, BM25, hash-semantic scoring and optional cross-encoder.
2. MariaDB/MySQL is selected by `ALITIGATOR_RAG_BACKEND=mysql` in the observed
   local runtime. `mysql_rag.py` mirrors most SQLite behavior and is the active
   retrieval backend.
3. Supabase has a separate RPC/storage implementation in `supabase_rag.py`.
   It is disabled locally. In `main.py` its search fallback is attempted only
   when the configured primary backend is SQLite and returned no chunks.
4. Corpus import/chunk preparation is backend-independent at the JSONL layer,
   but indexing/ranking logic is duplicated between SQLite and MySQL.

## Static routing and handcrafted rules

The primary baseline depends on several overlapping registries and detectors:

- `QUERY_EXPANSIONS`, `STATUTE_QUERY_EXPANSIONS`, statutory concepts,
  procedural rules and a mechanism lexicon in `rag.py`;
- `query_targets_*` functions for KSeF, housing relief, private vehicles,
  company transformations, family foundations, WHT, treaties, fixed
  establishment, debt assumption, real-estate sales and other benchmark-like
  mechanisms (`rag.py:2207-3776`);
- `build_*_statute_targets` and `preferred_targets` lists that nominate exact
  articles (`rag.py:2803-3638`, `6973-7210`, `8746-8825`);
- direct subject-prefix routing for selected acts/treaties;
- candidate bonuses/penalties for particular legal neighbors, source text,
  document IDs and article families (`rag.py:2970-3776`, `8196-8455`);
- a large global `SYSTEM_PROMPT` plus `build_chat_system_prompt`, containing
  rules for KSeF, family foundations, housing relief, WHT, vehicles,
  transformations, land sales, dropshipping, invoices and named provisions;
- heuristic intent, FactGraph, IssueGraph and clarifier logic in
  `hybrid_authority_rag.py:286-694`;
- heuristic AuthorityCard extraction based on regexes and selected spans in
  `hybrid_authority_rag.py:1196-1444`;
- special-pipeline regex routers and parsers in `bad_debt_pipeline.py`,
  `housing_relief_pipeline.py`, and `controlled_legal_pipeline.py`.

These rules are useful as regression oracles, calculators, validators and
fallback query hints. They are not a general model-driven planner.

## Hardcoded document routing

The following identifiers are explicitly used by baseline routing or direct
candidate boosting. This is a routing inventory, not a gold-label inventory.

Named bundles/constants near `rag.py:113-146`:

- KSeF interpretation: `679542`;
- KSeF deduction/correction neighborhood: `695345`, `695471`, `695355`,
  `695403`, `694097`, `693430`, `693595`, `693598`, `693253`, `693103`,
  `696243`, `696177`, `693053`, `694474`, `692135`, `695412`;
- debt assumption: `695395`, `678370`;
- temporary housing rental: `691376`;
- mortgage settlement: `688486`, `693529`;
- current KSeF bundle:
  `ksef-2-0-current-law-dzu-2025-1203-transition`,
  `ksef-2-0-offline24-operational-modes`,
  `ksef-2-0-scope-fixed-establishment-and-foreign-buyers`,
  `ksef-2-0-corrections-and-vat-deduction`;
- family-foundation primary bundle:
  `family-foundation-primary-ufr-art-5-27-29`,
  `family-foundation-primary-cit-24q-24r`,
  `family-foundation-primary-pit-beneficiary-rates`,
  `family-foundation-primary-vat-related-party-transactions`.

Additional inline direct-routing IDs in `search_chat_chunks`, candidate pool
construction and scoring include: `681556`, `683152`, `685154`, `685379`,
`685389`, `679544`, `687425`, `690463`, `691194`, `691352`, `691426`,
`692558`, `692562`, `692580`, `692665`, `693399`, `693582`, `694262`,
`694267`, `694316`, `694474`, `694510`, `694663`, `695099`, `695219`,
`695238`, `695345`, `695380`, `695412`, `695572`, and `696263`. Other
six-digit values occur in benchmark-oriented scoring tables; the complete
pre-refactor set is discoverable with:

```bash
rg -o '"[0-9]{6}"' apps/api/app/rag.py | sort -u
```

The pre-refactor file contains 95 unique six-digit EUREKA IDs and eight
synthetic bundle IDs. It also contains one complete signature used as a
selection signal: `0113-KDIPT1-2.4012.1035.2025.2.AJB` (`rag.py:2659`).

No such identifier may be imported or referenced by `app/legal_rag_v2`.

## Embeddings and granularity

- Index records label the current vector as `alitigator-hash-v1`.
  `compute_embedding` uses token/n-gram feature hashing. It is recomputed for
  query/candidates and is not a model embedding.
- BM25/FTS and an optional sentence-transformers cross-encoder remain valuable
  candidate-generation/reranking components.
- SQLite and MySQL do not persist the hash vectors; candidate vectors are
  recalculated during search. MySQL calculates an embedding payload while
  indexing and then discards it. Supabase is the only adapter that persists
  those vectors. Repeated high-weight fields do not actually gain weight,
  because the hash tokenizer deduplicates tokens before hashing.
- `law_chunk.py` emits one record per article (or size-based part split only at
  numbered paragraphs). It does not give every paragraph, point and letter a
  stable provision-unit identity.
- `treaty_chunk.py` emits article-level treaty records.
- No general persisted ProvisionGraph currently models references,
  definitions, exceptions, special rules, temporal successors, neighbors or
  transitional dependencies.
- SQLite creates `legal_document_versions` and `legal_provisions` tables, but
  current indexing does not populate them. MySQL search calls schema readiness
  code from the read path, so a nominal read can create/alter search tables.
- Hybrid `dependency_provisions` and `exception_provisions` are currently
  always empty. Its seven-year historical cutoff is a heuristic, and its
  normalized-text source spans do not reliably address the original chunk.

## Components worth preserving

- `legal_pipeline.py`: temporal provision registry, `LegalClaim`,
  `CalculationRecord`, claim validation and answer-plan primitives;
- controlled pipelines: deterministic calculators, regression oracles,
  provision registries and render validators (after removing them from v2
  routing);
- `RagChunk`, corpus normalization/import, SQLite/MySQL/Supabase access, BM25
  and full-text candidate generation;
- statute/treaty source extraction and official-source metadata;
- hybrid RAG's typed `AuthorityCard`, rerank reason structure,
  `AuthorityLine`, `EvidenceBundle` and artifact-writing concepts, after
  replacing their heuristic producer stages;
- existing temporal filters, source-type separation, citation formatting and
  deterministic post-render checks;
- `rag_eval.py`, `rag_law_eval.py` and the controlled test cases as baseline
  and development evaluation material.

## Baseline boundary for the refactor

The legacy files and their behavior must remain runnable. V2 should live beside
them, own its provider/planner/retrieval/claim/writer contract, contain no
hardcoded corpus document IDs, and be selected only through an explicit
`LEGAL_PIPELINE_MODE` flag. The current holdout was not opened, edited or run
during this audit.
