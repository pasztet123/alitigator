# Current Architecture Audit

## Retrieval Entry Points

- Chat retrieval is built in `apps/api/app/main.py` inside `/api/chat`. The effective query is the latest user message plus any UI intent hints from `build_effective_user_prompt`.
- The main retrieval call is `search_chat_chunks(effective_user_prompt, include_interpretations=..., include_judgments=...)` from `apps/api/app/rag.py`.
- The raw `/api/rag/search` endpoint uses `inspect_search` and `search_chunks`; it is useful for retrieval diagnostics but is not the same as chat retrieval.
- Current benchmark runners are split between:
  - `app.rag_eval` for interpretation/judgment/document retrieval cases with `expected_document_ids` and `expected_signatures`.
  - `app.rag_law_eval` for primary-law retrieval cases with `expected_legal_provisions` and optional holdout exclusion.
  - shell wrappers under `apps/api/scripts/run_*_benchmark.sh`, where VAT and nieruchomosci wrappers explicitly exclude holdout files.

## Current Query Planning

- `search_chat_chunks` builds a `LegalSourcePlan` via `build_legal_source_plan`.
- `build_legal_source_plan` decomposes specialist topics into `LegalRetrievalAxis` objects using `decompose_query_into_legal_axes`, detects explicit and preferred statute article targets, and sets primary source types to `("statute",)`.
- Interpretations and judgments are currently routed by source type budgets. They are not searched by a separate authority evidence model; they are retrieved as secondary chunks and mixed with primary-law chunks.
- The baseline does not use one single untyped query in chat mode. It separately retrieves statutes, interpretations, and, when requested or inferred, judgments. However, each secondary lane still mostly uses the same effective natural-language query plus existing hard-coded specialist expanders.

## Indexed Document Types

Local SQLite table `documents` currently contains:

- `interpretation` / `individual`: 10103 documents.
- `judgment` / `nsa`: 2365 documents.
- `statute` / `consolidated_text`: 2751 documents.
- `statute` / `codified_text`: 418 documents.
- `statute` / `tax_treaty`: 351 documents.
- small statute bundles for family foundation and KSeF guidance.

There are no WSA judgments in the local index observed during this audit; the judgment corpus is currently NSA.

## Metadata Available

The SQLite `documents` table has:

- `document_id`, `content_sha256`, `subject`, `signature`, `published_date`, `source_url`, `category`;
- `keywords_json`, `legal_provisions_json`, `issues_json`, `law_tags_json`;
- `tax_domain`, `signature_family`, `question_text`, `decision_text`, `facts_text`;
- `source`, `source_type`, `source_subtype`, `authority`, `jurisdiction`;
- `act_title`, `publication`, `legal_state_date`, `source_pages_json`.

This is enough to test document type, tax domain, authority/court, signatures, dates, cited provisions, fact text, and source spans at chunk level. It is not yet enough to know every authority holding without extraction.

## Grouping And Context

- Retrieval returns `RagChunk` objects.
- `select_diverse_chunks` and document-context helpers group by canonical source/document when building context.
- `build_answer_context_block` may expand selected chunks to full document context through `fetch_document_contexts` when document context is enabled.
- `list_citations` deduplicates by `chunk_canonical_source_id`.

## Reranking

- Candidate generation is FTS plus direct/hard-coded channels in `fetch_local_candidate_rows`.
- `rank_hybrid_local_candidates` combines lexical, semantic, legal match, mechanism match, judgment match, and several specialist heuristics.
- `rerank_chunks_within_documents` fetches more chunks from already-selected documents and reranks the best fragment per document.
- There is no explicit `AuthorityCard` reranker that scores legal similarity across taxpayer role, transaction type, payment type, holding, temporal status, and wrong-neighbor reasons.

## Legal Claim Pipeline

- Primary-law chunks are converted to `LegalRule` by `extract_legal_rules_from_statute_chunks`.
- `build_registry_from_rules` builds a `ProvisionRegistry`.
- `build_claims_from_rules` builds `LegalClaim` objects from extracted primary-law rules.
- `validate_claim` checks temporal applicability, fact dependencies, calculations, tax domain, taxpayer role, mechanism, and special-rule conflicts.
- `build_analysis_trace` exposes claim traces and selected documents.

## Where Authorities Disappear

- Interpretations and judgments reach the final writer as source context and citations.
- They do not currently become first-class claim evidence in `LegalClaim`.
- `LegalClaim` currently has controlling/dependency provisions and provenance, but no structured `supporting_authorities`, `contrary_authorities`, `historical_authorities`, or authority confidence fields.
- Therefore secondary authorities currently have no validated role comparable to primary-law provisions. They are neither controlling sources nor structured supporting/contrary sources in the claim set.
- The current prompt tells the writer to distinguish statutes, interpretations, and judgments, but this is prompt-level discipline rather than validated claim binding.

## Taxpayer Versus Authority Position

- The baseline has section-role heuristics such as `classify_chunk_evidence_role`, and prompt rules warn not to treat taxpayer facts or positions as holdings.
- There is no cached extraction object that separately stores `taxpayer_position`, `authority_holding`, `court_holding`, `outcome`, and source spans.

## Extraction Cache

- The RAG index stores chunks and document metadata.
- No dedicated `AuthorityCard` cache was found. A suitable cache key for the experiment should include document hash, extractor model, extractor prompt version, and schema version.

## Benchmark Format

- `app.rag_eval` expects each case to have `id`, `question`, and at least one of `expected_document_ids` or `expected_signatures`.
- `app.rag_law_eval` expects `id`, `question`, and `expected_legal_provisions`; it supports `--exclude-cases` and should be used only with dev/seed cases for this experiment.
- `expected_signatures` are matched against document signatures in the local RAG index and are the natural basis for authority recall metrics.

## Experiment Hook Points

- Keep `search_chat_chunks` as baseline A.
- Add a feature-flagged alternative around the chat retrieval section in `main.py`.
- Reuse `LegalRetrievalAxis`, `LegalSourcePlan`, `RagChunk`, `inspect_search`, `search_chunks`, and legal rule extraction.
- Add structured authority retrieval and reranking before existing `LegalClaim` validation, then bind selected authority evidence into expanded `LegalClaim` fields.
