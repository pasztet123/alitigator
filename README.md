# aLitigator MVP

Minimalne MVP platformy do researchu podatkowego i roboczego pisania pism.

## Stack

- frontend: React + Vite
- backend: FastAPI
- model: Claude API przez backendowy proxy
- baza danych: Supabase w kolejnym kroku

## Uruchomienie lokalne

### Frontend

```bash
cd apps/web
cp .env.example .env.local
npm install
npm run dev
```

### Backend

```bash
cd apps/api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

## Sekrety

Nie zapisuj kluczy API w repo. Backend czyta `ANTHROPIC_API_KEY` z pliku `.env` albo ze zmiennych środowiskowych.

## Status

Ta wersja zawiera:

- prosty czat w zielonej identyfikacji aLitigator
- backendowy proxy do Claude
- podstawowe maskowanie danych wrażliwych
- miejsce na późniejsze podpięcie Supabase, auth, kredytów i RAG

## Import interpretacji do RAG

Backend ma importer interpretacji z EUREKA, który zapisuje dwa pliki JSONL: surowy payload (`raw`) i znormalizowany dataset (`processed`).

CLI:

```bash
cd apps/api
. .venv/bin/activate
python -m app.eureka_ingest --limit 1000
```

Stabilniejsze ustawienia dla szerokiego importu:

```bash
python -m app.eureka_ingest --limit 1000 --page-size 20 --concurrency 1 --request-timeout 45
```

Domyślne pliki wyjściowe:

- `apps/api/data/raw/eureka_interpretations.raw.jsonl`
- `apps/api/data/processed/eureka_interpretations.jsonl`

Importer wspiera wznowienie przez `--start-page`, append bez duplikatów oraz filtrowanie po `--law-tag '[CIT]'`.

Można też uruchomić import przez API:

```bash
curl -X POST http://127.0.0.1:8000/api/rag/eureka/import \
	-H 'Content-Type: application/json' \
	-d '{"limit":1000}'
```

## Lokalny RAG na interpretacjach

Backend ma teraz dwa tryby indeksu RAG:

- domyślny: lokalny SQLite FTS jako główne źródło retrievalu,
- opcjonalny: Supabase, jeśli chcesz świadomie przenieść corpus do zdalnej bazy.

Domyślny przepływ jest taki:

- importer dalej dopisuje nowe interpretacje do `processed` JSONL jako lokalnego bufora roboczego,
- `POST /api/rag/reindex` domyślnie buduje lokalny indeks SQLite,
- `/api/chat` korzysta z lokalnego indeksu, a Supabase jest tylko opcjonalnym rozszerzeniem.

Zbudowanie albo odświeżenie indeksu:

```bash
cd apps/api
. .venv/bin/activate
PYTHONPATH=/Users/stas/alitigator/apps/api python -m uvicorn app.main:app --reload --port 8000
```

W drugim terminalu:

```bash
curl -X POST http://127.0.0.1:8000/api/rag/reindex \
	-H 'Content-Type: application/json' \
	-d '{}'
```

Przykładowa reindeksacja wymuszona tylko dla próbki:

```bash
curl -X POST http://127.0.0.1:8000/api/rag/reindex \
	-H 'Content-Type: application/json' \
	-d '{"limit":200,"force":true}'
```

Lokalny fallback zapisuje indeks w:

- `apps/api/data/processed/eureka_rag.sqlite3`

### Diagnostyka retrievalu

Żeby oceniać retrieval bez modelu, backend udostępnia endpoint diagnostyczny:

```bash
curl -X POST http://127.0.0.1:8000/api/rag/search \
	-H 'Content-Type: application/json' \
	-d '{"query":"Czy przysługuje prawo do odliczenia VAT od wydatków na realizację inwestycji?"}'
```

W odpowiedzi dostajesz:

- surowe trafienia z rankingiem,
- informację, które chunky weszły do kontekstu,
- gotowy `context_block`,
- cytowania do szybkiej oceny jakości retrievalu.

### Lokalny evaluator RAG

Możesz odpalać serię pytań testowych bez modelu:

```bash
cd apps/api
PYTHONPATH=/Users/stas/alitigator/apps/api .venv/bin/python -m app.rag_eval
```

Domyślnie evaluator czyta plik:

- `apps/api/data/processed/rag_eval_cases.sample.json`

Własny zestaw pytań możesz przekazać tak:

```bash
PYTHONPATH=/Users/stas/alitigator/apps/api .venv/bin/python -m app.rag_eval --cases /sciezka/do/cases.json --fail-on-miss
```

Format pojedynczego case'a:

```json
{
	"id": "krotki-identyfikator",
	"question": "Pytanie użytkownika",
	"expected_document_ids": ["opcjonalne-document-id"],
	"expected_signatures": ["opcjonalna-sygnatura"],
	"notes": "opcjonalny komentarz",
	"expected_answer": "krótka wzorcowa odpowiedź dla człowieka"
}
```

Konfiguracja chunkingu i retrievalu jest sterowana przez zmienne `ALITIGATOR_RAG_*` w `apps/api/.env.example`.

Retriever rozwija też podstawowe skróty podatkowe (np. `KSeF`, `WHT`, `PCC`, PSH) do ich pełnych nazw. Raport evaluatora zawiera `expected_in_raw_candidate_pool` i `lost_in_rerank`, co pozwala odróżnić brak recall od błędnego kolejnościowania przez reranker. Po włączeniu trybu Supabase wykonaj ponowny backfill, aby zapisać nowe, mocniej ważone embeddingi chunków.

### Opcjonalny storage w Supabase

Jeżeli chcesz świadomie pracować na zdalnym corpusie, najpierw utwórz tabele i funkcję wyszukiwania z pliku:

- `apps/api/sql/rag_schema.sql`

Następnie ustaw:

- `SUPABASE_URL`
- `SUPABASE_SECRET_KEY`
- `ALITIGATOR_RAG_SUPABASE_SYNC=true`
- `ALITIGATOR_RAG_USE_SUPABASE=true`

albo wywołuj reindeksację z `{"sync_supabase": true}`.

Jeżeli chcesz robić backfill bez odpalania endpointu i bez ręcznej obsługi, używaj lokalnego skryptu CLI:

```bash
cd apps/api
. .venv/bin/activate
python -m app.supabase_backfill --status
python -m app.supabase_backfill
```

Przydatne warianty:

```bash
python -m app.supabase_backfill --limit 200
python -m app.supabase_backfill --reset-state
python -m app.supabase_backfill --force
```

Skrypt czyta checkpoint z `apps/api/data/processed/eureka_supabase_sync_state.json`, więc można go bezpiecznie wznawiać po przerwaniu.

W tym trybie corpus RAG siedzi w Supabase:

- pełne rekordy interpretacji w `public.eureka_interpretations`,
- chunki do retrievalu w `public.eureka_chunks`,
- backend FastAPI tylko zasila tabele, odpytuje retrieval i składa prompt.

Obecna wersja nadal zostawia JSONL jako roboczy bufor ingestu, ale ten tryb jest opcjonalny i nie jest już domyślną ścieżką runtime.

## Supabase

- schemat RAG jest przygotowany pod `public`, żeby działał od razu z domyślną ekspozycją Data API
- tabele bazowe: `profiles`, `credit_ledger`, `chat_threads`, `chat_messages`
- RLS jest włączone na wszystkich czterech tabelach
- sekret `SUPABASE_SECRET_KEY` ma pozostać wyłącznie po stronie backendu
