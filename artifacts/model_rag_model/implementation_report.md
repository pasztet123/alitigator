# Raport implementacji Model → RAG → Model

Data: 2026-07-13. Holdout nie został otwarty, uruchomiony ani zmodyfikowany.

## Zakres

- Zachowano `legacy`; dodano publiczne tryby `model_rag_model` i `shadow` oraz
  alias kompatybilności `legal_rag_v2`.
- Planner modelowy działa przed retrieval; `LegacyFallbackPlanner` jest wyłącznie
  śledzonym fallbackiem technicznym/low-confidence.
- Primary i authority lanes uruchamiają się per issue. Interpretacje i wyroki są
  domyślnie dostępne niezależnie od słów „wyrok”, „NSA” albo „WSA”.
- Retrieval łączy FTS, opcjonalne real embeddings, metadata, provision references
  i dwukierunkowe backreferences; limit iteracji wynosi dwa.
- Chunking i ProvisionGraph zachowują article/paragraph/point/letter oraz daty
  obowiązywania. Nowa ścieżka nie używa `chunk_index = 0` jako direct lookup.
- Evidence jest modelowane przez exact-span AuthorityCards, legal reranking,
  wrong-neighbor rejection, source→issue i source→claim bindings.
- Materialne claimy bez primary law, claimy authority bez authority ID i claimy
  kalkulacyjne bez valid CalculationRecord są blokowane.
- Writer działa po claim validation; źródła są budowane z registry, a final
  validator działa fail-closed.
- Trace zawiera wszystkie wymagane 38 artefaktów, w tym obie lineage, token/cost
  status, reranking, rejections, bindings i drugą iterację.

## Nowe moduły

`app/legal_research`: `config`, `models`, `tracing`, provider gateways, planner i
fallback, `retrieval/*`, `evidence/*`, `claims/*`, `answer/*`, `pipeline`.

## Routing i early returns

Nie usunięto early returns legacy. W trybie `model_rag_model` wspólny pipeline
jest wywoływany przed benchmark bad-debt, bad-debt, mixed-invoice i
housing-relief, więc żaden z nich nie może ominąć nowej ścieżki. W `legacy`
zachowują dotychczasowe zachowanie.

## Reguły przeniesione/izolowane jako fallback

`classify_legal_research_intent`, `build_fact_graph`, `build_issue_graph` oraz
`heuristic_clarifier_questions` są ładowane leniwie przez
`LegacyFallbackPlanner`. Ręczne `query_targets_*`, query expansion, wymagane
document IDs i benchmark labels nie są importowane przez główny planner ani
neutralny backend nowego pipeline'u.

Hardcodowane document IDs w `app/legal_research`: **0**.

## Walidacja

- Backend: **142/142 testów** przeszło.
- Frontend: `npm run build` przeszedł.
- Compile/import i `git diff --check`: przeszły.
- Dodano kontrasty dla source lineage, incomplete holding, niezależnych progów
  abstention oraz granic deadline: dzień przed, dokładnie w terminie, dzień po.

## A/B/C — jawny smoke, jeden przypadek developerski

To pomiar integracyjny, nie reprezentatywny benchmark jakości.

| Wariant | authority recall@5 | recall@20 | latency | fallback | wynik |
|---|---:|---:|---:|---:|---|
| A legacy | 0.00 | 1.00 | 74.3 s | 0% | retrieval-only |
| B model_rag_model | 0.00 | 0.00 | 84.6 s | 100% | partial, fail-closed |
| C z fallbackiem | 0.00 | 0.00 | 67.8 s | 100% | partial, fail-closed |

B wykrył niedostępność/niezgodność skonfigurowanego providera Structured
Outputs i przeszedł do audytowalnego fallbacku `provider_unavailable`. Nie
powstał żaden unsupported material claim; approved claims bez primary source =
0, false authority claims = 0, blank legal references = 0. Druga iteracja
retrievalu wykonała się w B i C. Authority source-span coverage wyniosło 87,5%.
Claim validation celowo zablokowało pełną odpowiedź.

Szczegółowe wyniki: `ab-smoke/comparison.json` i
`ab-smoke-bc/comparison.json`; pełne trace znajdują się w `ab-smoke-bc/runs`.

## Metryki i ograniczenia

- wrong-neighbor rate i authority abstention rate są zapisywane per run w
  `metrics.json`; smoke sprzed ostatniego rozszerzenia metryk nie jest używany do
  fabrykowania wartości.
- second retrieval rate w smoke: 100% dla B/C.
- fallback rate w smoke: 100% dla B/C (awaria/niezgodność providera).
- Koszt per request: niedostępny, ponieważ aktualny gateway nie udostępnia usage;
  `token_usage.json` i `costs.json` mówią to jawnie zamiast estymować.
- Największe ryzyka: niedziałający obecnie modelowy provider dla structured
  pipeline'u, wysoka latency oraz recall authority wymagający pełnego dev runu.

## Rekomendacja produkcyjna

Wdrożyć kod 2.0.0 z `LEGAL_RAG_MODE=legacy`, zachowując natychmiastowy rollback
i możliwość kontrolowanego `shadow`. Nie przełączać produkcyjnych odpowiedzi na
`model_rag_model`, dopóki provider nie przejdzie end-to-end bez fallbacku i
pełny jawny benchmark nie potwierdzi jakości, kosztu oraz latency.
