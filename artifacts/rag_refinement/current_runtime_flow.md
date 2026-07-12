# Aktualny runtime flow RAG

## Konfiguracja i routing

`LEGAL_RAG_MODE` jest publiczną flagą runtime z wartościami `legacy`, `rag_v2`
i `shadow`. Dla zgodności odczytywana jest też wcześniejsza
`LEGAL_PIPELINE_MODE` (`legal_rag_v2` jest jej aliasem dla `rag_v2`).

- `legacy`: `POST /api/chat` wykonuje istniejące routery controlled pipeline,
  następnie baseline albo `LEGAL_RETRIEVAL_MODE=hybrid_authority`; answer model
  pozostaje dotychczasowym writerem.
- `rag_v2`: request trafia przed routerami controlled do
  `LegalRagV2Pipeline`; controlled pipeline nie może wybrać źródła ani
  wygenerować odpowiedzi.
- `shadow`: użytkownik dostaje legacy; niezależny task uruchamia v2 i zapisuje
  jego trace. Wynik legacy nie jest wejściem v2.

Backend legacy wybiera `rag.py`/`mysql_rag.py` na podstawie
`ALITIGATOR_RAG_BACKEND`; Supabase jest osobnym, opcjonalnym fallbackiem.
V2 używa neutralnego `CorpusFtsBackend`, opcjonalnego versioned vector indexu
i nie uruchamia ręcznych routerów produkcyjnych.

## Legacy

```text
chat -> controlled special-case routers -> baseline/hybrid retrieval
     -> static rules/source plan -> direct writer -> guardrails
```

W baseline primary-law failure może ograniczyć dalszy kontekst. Hybrid ma
heurystyczny planner i AuthorityCard. Odpowiedź legacy nie jest gate’owana
przez claim-level EvidenceBundle v2.

## RAG v2

```text
chat -> model LegalQueryPlanner (Structured Outputs)
     -> per-issue primary + authority lanes (zawsze oba)
     -> authority citations -> retry primary (iteracja 1)
     -> primary citations -> retry authority (iteracja 2)
     -> model AuthorityCard -> issue-scoped EvidenceBundle
     -> claim synthesis + deterministic validation -> constrained writer
     -> deterministic renderer + post-render validation
```

Planner fallback (`LegacyFallbackPlanner`) jest dozwolony wyłącznie po
timeoutcie, błędzie providera/schemy albo niskiej pewności i zostawia ślad w
trace. Brak controlling provision blokuje materialny claim; writer dostaje
wyłącznie zatwierdzone claimy, bundle’y i registry referencji.

Każdy run v2 zapisuje `runtime.json` z: pipeline/retrieval/backend/planner/
extractor/answer model, informacją o controlled pipeline oraz fallbackach.
Trace obejmuje też candidate pools, backreferences, coverage, lineage,
claims, writer payload/output, walidację i metryki.
