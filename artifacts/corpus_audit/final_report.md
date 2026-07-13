# RAG corpus audit

## Root cause

The production read backend is MySQL. Its inventory has statutes and
interpretations, but no documents with `source_type=judgment`. The local
source corpus is complete: it contains the configured interpretation file,
all configured statutory/treaty bundles and a judgment corpus. This is an
index/backfill completeness failure, not a missing-source-data failure.

The audit also found three contributing defects:

1. the API image did not copy `data/`, preventing a container-side reindex;
2. Supabase backfill used only the main interpretation file and a global v1
   cursor, so newly configured sources could be skipped;
3. the Supabase adapter/RPC did not pass source filters or reconstruct full
   document metadata. The public search endpoint additionally treated the
   abbreviated MySQL inspection hit as a complete API hit.

## Changes made locally

- Added a shared `resolve_rag_runtime()` and explicit `RAG_READ_BACKEND`,
  `RAG_WRITE_BACKEND`, `RAG_FALLBACK_BACKEND` routing. There is no implicit
  Supabase fallback.
- Added one shared source manifest for SQLite, MySQL and Supabase reindexing.
- Made mixed professional retrieval include judgments unless explicitly
  disabled.
- Upgraded the Supabase resumable state to v2 with per-source cursors and a
  manifest hash; backfill remains idempotent upsert-only.
- Added additive Supabase metadata/RPC SQL and full `RagChunk` reconstruction.
- Added read-only corpus diagnostics, an admin-only corpus health endpoint,
  inventory artifacts, and an optional `RAG_REQUIRE_COMPLETE_CORPUS` gate.
- Kept the multi-gigabyte source corpus out of the runtime image and added a
  local/controlled backfill path to MySQL; fixed `/api/rag/search` hit
  serialization.

## Verification

- Local corpus inventory: valid configured source files include statutes,
  interpretations and judgments.
- Active MySQL direct retrieval: statute and interpretation filters return
  results; judgment filter returns zero results.
- Local API smoke using the active MySQL configuration: statute and
  interpretation endpoint paths return typed hits; judgment returns an empty
  result without an endpoint error.
- Live production endpoint returned no judgment hits and returned HTTP 500 for
  statute/interpretation searches because of the now-fixed hit-serialization
  defect. No production reindex, migration, deletion, commit or deployment
  was run.

**FIX VERIFIED END TO END: NO.** A production deployment followed by a
non-destructive MySQL reindex/backfill is still required. Only then can an
actual production endpoint be verified to return a statute, interpretation
and judgment.
