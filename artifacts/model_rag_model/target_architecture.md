# Docelowy przepływ Model → RAG → Model

```text
POST /api/chat
→ LEGAL_RAG_MODE
   ├─ legacy: niezmieniona ścieżka produkcyjna
   ├─ shadow: odpowiedź legacy + niezależny trace nowego pipeline'u
   └─ model_rag_model
      → LegalResearchPlanner (Structured Outputs)
      → ResearchPlan: facts, roles, issues, constraints, missing facts
      → PrimaryLawLane + AuthorityLane osobno dla każdego issue
      → szeroki candidate pool: FTS + real vectors + metadata + references
      → jawny legal reranker i wrong-neighbor penalties
      → Evidence Analyst: reguły, AuthorityCards, binding, missing evidence
      → maksymalnie druga iteracja authority ↔ primary
      → temporal ProvisionGraph + EvidenceBundle per issue
      → deterministyczne LegalClaims i CalculationRecords
      → fail-closed claim validation
      → constrained Answer Writer
      → deterministyczne źródła i renderer
      → final validator + exact reference lineage
      → odpowiedź albo kontrolowane zablokowanie
```

Reguły `query_targets_*`, ręczne ID dokumentów i specjalne routery pozostają
wyłącznie w `legacy`. `LegacyFallbackPlanner` jest wywoływany po błędzie
technicznym, timeoutcie, invalid schema lub niskiej confidence; nie ma API do
tworzenia finalnej odpowiedzi.

Publiczny pakiet `app/legal_research` wyznacza granice: config i modele,
provider-neutral gateway, planner, retrieval, evidence, claims/calculations,
answer i tracing. Komponenty z pierwszej iteracji `legal_rag_v2` są pod nim
adaptowane stopniowo, bez kopiowania monolitycznego `rag.py`.
