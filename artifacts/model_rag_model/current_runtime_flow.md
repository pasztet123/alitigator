# Audyt rzeczywistego runtime flow Alitigatora

Data audytu: 2026-07-13. Zakres obejmuje produkcyjny endpoint `POST /api/chat`,
warstwy RAG, pipeline'y kontrolowane, backendy oraz runnery developerskie. Audyt
nie otwierał ani nie uruchamiał holdoutu.

## Główny routing HTTP

```text
POST /api/chat
→ uwierzytelnienie, redakcja PII, zapis wiadomości i zbudowanie effective_user_prompt
→ LEGAL_RAG_MODE / LEGAL_PIPELINE_MODE
   ├─ legal_rag_v2: LegalRagV2Pipeline (przed wszystkimi routerami legacy)
   ├─ shadow: niezależny task LegalRagV2Pipeline + dalsza odpowiedź legacy
   └─ legacy: dalszy routing specjalny i standardowy retrieval
→ zapis odpowiedzi / pobranie kredytu / ChatResponse
```

W stanie sprzed refaktoryzacji publicznie obsługiwane wartości to `legacy`,
`legal_rag_v2` (oraz alias `rag_v2`) i `shadow`. Wymagany przez nową
specyfikację kontrakt `model_rag_model` nie jest jeszcze obsługiwany.

## Ścieżka legacy

```text
HTTP request
→ benchmarkowy early return ulgi na złe długi (jeżeli payload ma marker trace)
→ kontrolowany pipeline ulgi na złe długi (jeżeli regex + komplet faktów)
→ kontrolowany pipeline faktur mieszanych (jeżeli regex)
→ kontrolowany pipeline ulgi mieszkaniowej (jeżeli regex + komplet faktów)
→ LEGAL_RETRIEVAL_MODE
   ├─ hybrid_authority: heurystyczny intent/fact/issue graph
   │  → primary + authority retrieval → AuthorityCard → context
   └─ baseline: rag.py / mysql_rag.py
      → query_targets_* → ręczne query expansion / wymagane dokumenty
      → statuty + interpretacje + orzeczenia
→ model odpowiedzi
→ postprocessing i source guardrails w main.py
→ odpowiedź
```

Alternatywne early returns w `main.py` znajdują się przed ogólnym retrieval i
writerem. Są to: benchmark bad-debt, produkcyjny bad-debt pipeline, mixed
invoice pipeline i housing-relief pipeline. Każdy buduje własne fakty, claimy,
kalkulacje i renderer. W trybie v2 są omijane, ponieważ wspólny pipeline zwraca
odpowiedź wcześniej.

## Istniejąca ścieżka legal_rag_v2

```text
HTTP request
→ modelowy planner Structured Outputs
→ LegalResearchPlan z faktami, issues i query families
→ dla każdego issue osobno:
   ├─ primary-law lane
   └─ authority lane (interpretacje i orzeczenia)
→ lexical/vector fusion + jawny reranking
→ ekstrakcja AuthorityCard i exact spans
→ backreference authority → primary i primary → authority (maks. 2 przebiegi)
→ ProvisionGraph + temporal filtering
→ EvidenceBundle per issue
→ structured LegalClaims + deterministyczna walidacja
→ AnswerPlan → constrained writer
→ deterministyczny renderer → post-render validation
→ trace i ChatResponse
```

`shadow` uruchamia tę ścieżkę w osobnym tasku i nie przekazuje jej odpowiedzi
legacy. Odpowiedź użytkownika nadal powstaje całkowicie w ścieżce legacy.

## Backendy retrievalu

- Legacy SQLite: `rag.py`, lokalne FTS i indeks dokumentów.
- Legacy MySQL: `mysql_rag.py`, wybierany przez `ALITIGATOR_RAG_BACKEND`.
- Supabase: `supabase_rag.py`, synchronizacja/backfill i opcjonalne zdalne dane.
- Hybrid authority: `hybrid_authority_rag.py`, nakładka z heurystycznym planem,
  dwiema liniami źródeł i ekstrakcją kart.
- V2: neutralny `CorpusFtsBackend`, opcjonalny osobny indeks embeddingów oraz
  jawny offline hash fallback. Backend v2 nie importuje `query_targets_*`.

## Feature flags i modele

- `LEGAL_RAG_MODE` / kompatybilne `LEGAL_PIPELINE_MODE`: wybór pipeline'u.
- `LEGAL_RETRIEVAL_MODE`: baseline lub hybrid w legacy.
- `ALITIGATOR_RAG_BACKEND`: SQLite/MySQL dla legacy i korpusu.
- `LEGAL_RAG_V2_ALLOW_LEGACY_FALLBACK`: reguły legacy wyłącznie po błędzie,
  invalid schema, timeoutcie lub niskiej confidence plannera.
- `LEGAL_RAG_V2_REQUIRE_REAL_EMBEDDINGS` i
  `LEGAL_RAG_V2_ALLOW_OFFLINE_HASH_EMBEDDINGS`: polityka wektorów.
- `LLM_PROVIDER`, stage-specific modele i gateway fallback: wybór providera oraz
  modeli plannera, analityka/syntezy i writera.

Wywołania modeli legacy znajdują się w `main.py` i ścieżkach ekstrakcji.
Provider-neutral boundary znajduje się w `model_gateway.py`; istniejące v2
wywołuje model przed retrieval (planner), po retrieval (authority/evidence) i
po walidacji claimów (writer).

## Claimy, obliczenia, filtrowanie i postprocessing

- Legacy claimy oraz kalkulatory są rozproszone między `legal_pipeline.py`,
  `controlled_legal_pipeline.py`, `housing_relief_pipeline.py` i
  `bad_debt_pipeline.py`.
- V2 buduje bundle'y i claimy centralnie w `legal_rag_v2/pipeline.py`; materialny
  claim bez primary law jest blokowany, a kalkulacja wymaga `CalculationRecord`.
- Legacy filtruje źródła w `rag.py`, `mysql_rag.py`, hybrid rerankerze oraz w
  `main.py`. W wielu miejscach szeroki topic match, wymagane ID albo limit
  źródeł może usunąć secondary source.
- Legacy postprocessing w `main.py` naprawia/usuwa referencje już po modelu,
  włącznie z fallbackiem tekstowym „ten przepis”. V2 waliduje writer output i
  blokuje odpowiedź zamiast destrukcyjnie modyfikować gotową treść.

## Hardcody i case-specific expansion

`rag.py` i `mysql_rag.py` zawierają liczne `query_targets_*`, zestawy wymaganych
document IDs, bezpośrednie ID interpretacji i ręczne query expansion. Pozostają
one w ścieżce legacy oraz jako jawnie audytowalny fallback/oracle. Nowy backend
v2 nie ma wpisanych identyfikatorów dokumentów ani expected signatures.

Legacy direct article retrieval w `hybrid_authority_rag.py` i `mysql_rag.py`
zawiera zapytania z `chunk_index = 0`. Strukturalny chunking w `law_chunk.py` i
`treaty_chunk.py` zapisuje article/paragraph/point/letter, a v2 używa stabilnego
`provision_id`, więc nie dziedziczy ograniczenia pierwszego chunka.

Gold labels (`expected_document_ids`, `expected_signatures`, expected
provisions) występują w evaluatorach developerskich. Są używane po retrieval
do pomiaru, nie są wejściem planera ani nowego retrievalu.

## Gdzie mogą zniknąć secondary sources

1. Legacy intent może nie włączyć właściwego source type.
2. Case-specific candidate generation może ograniczyć pulę do ręcznego bundle'a.
3. Topic/phrase/provision reranking i limity per typ mogą odrzucić authority.
4. Wspólny document-context limit może usunąć niżej sklasyfikowane dokumenty.
5. Source filtering i postprocessing `main.py` mogą odrzucić lub przepisać
   referencję po wygenerowaniu odpowiedzi.
6. Specjalne pipeline'y mogą wrócić przed ogólnym authority retrieval.

V2 uruchamia authority lane per issue niezależnie od sukcesu primary lane,
zapisuje przyczynę abstention i nie usuwa authorities tylko z powodu częściowego
primary coverage.

## Gdzie model może odpowiedzieć bez primary law

- W legacy ogólny writer może dostać niepełny kontekst po nieudanym primary
  retrieval i uzupełnić normę z własnej pamięci.
- Hybrid legacy przygotowuje lepszy kontekst, lecz nie ma jednolitego,
  fail-closed claim gate dla każdej materialnej konkluzji.
- Specjalne pipeline'y zwykle mają statyczny verified bundle, ale jest to osobna
  ścieżka i nie generalizuje.
- V2 dopuszcza writer dopiero po walidacji claimów; brak controlling provision
  daje status blokujący i nie może stać się zatwierdzoną konkluzją normatywną.

## Pierwszy etap odpowiedzialny za błąd

Istniejący trace v2 zapisuje plan, queries, candidate pools, reranking,
backreferences, graph, bundle'y, claimy, writer payload/output, validation oraz
lineage. Pozwala to przypisać błąd do planowania, candidate generation,
selekcji/fragmentu, temporal resolution, ekstrakcji authority, bindingu,
claim validation, writera, renderera albo final validation. Legacy nie zapewnia
równoważnej, jednolitej obserwowalności dla wszystkich early returns.

## Wniosek audytu

Najbezpieczniejsza droga migracji to zachowanie legacy, wykorzystanie
przetestowanych neutralnych komponentów v2 za nowym publicznym pakietem
`app/legal_research`, dodanie dokładnego kontraktu `model_rag_model` oraz
uzupełnienie schematów, trace i walidatorów bez ponownego włączania reguł
case-specific do głównej ścieżki.
