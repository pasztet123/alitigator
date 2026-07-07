# aLitigator MVP

Minimalne MVP platformy do researchu podatkowego i roboczego pisania pism.

## Stack

- frontend: React + Vite
- backend: FastAPI
- model: Claude API przez backendowy proxy
- baza danych kont i billing: Supabase
- baza dokumentów RAG: lokalny SQLite albo zewnętrzny MariaDB/MySQL

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

- logowanie i rejestrację przez Supabase Auth
- profile użytkowników i historię wątków przypisaną do `user_id`
- ledger kredytów po stronie backendu
- przygotowany checkout Stripe do sprzedaży doładowań kredytów
- backendowy proxy do Claude
- podstawowe maskowanie danych wrażliwych
- RAG lokalny i opcjonalny storage w Supabase

## Wersjonowanie

Przyjmujemy prostą zasadę numeracji wersji:

- małe zmiany podbijają wersję o `0.0.1`
- większe zmiany podbijają wersję o `0.1.0`
- major update podbija wersję o `1.0.0`

## Konta, kredyty i Stripe

Nowy schemat pod konta i billing znajduje się w:

- `apps/api/sql/auth_billing_schema.sql`

Schemat dodaje:

- `profiles`
- `credit_ledger`
- `credit_orders`
- rozszerzenie `chat_threads` o `user_id`
- polityki RLS dla profili, ledgera, zamówień i czatów

Backend udostępnia teraz:

- `GET /api/account`
- `PATCH /api/account/profile`
- `POST /api/billing/checkout-session`
- `POST /api/billing/webhooks/stripe`
- `POST /api/admin/credits/grant`

Ważne założenie MVP:

- sprzedajemy pakiety kredytów, a backend rozlicza każde zapytanie stałą stawką `ALITIGATOR_CREDIT_COST_PER_QUERY`
- to jest warstwa billingowa gotowa pod Stripe i konto użytkownika; dokładniejsze meteringi można dołożyć później bez przebudowy podstawowego flow

### Konfiguracja Supabase i Stripe

1. W Supabase uruchom SQL z:
   - `apps/api/sql/auth_billing_schema.sql`
2. Ustaw backendowe sekrety w `apps/api/.env`:
   - `SUPABASE_URL`
   - `SUPABASE_SECRET_KEY`
   - opcjonalnie `ALITIGATOR_ADMIN_EMAILS=admin@twojadomena.pl` jako jednorazowy bootstrap `profiles.is_admin`
   - `STRIPE_SECRET_KEY`
   - `STRIPE_WEBHOOK_SECRET`
   - `ALITIGATOR_STRIPE_SUCCESS_URL`
   - `ALITIGATOR_STRIPE_CANCEL_URL`
3. Ustaw frontend:
   - `VITE_SUPABASE_URL`
   - `VITE_SUPABASE_PUBLISHABLE_KEY`
4. Ustaw cenę kredytów:
   - `ALITIGATOR_CREDIT_UNIT_PRICE_GROSS` (domyślnie `200`, czyli 2,00 zł za 1 kredyt)
   - `ALITIGATOR_CREDIT_CURRENCY` (domyślnie `pln`)
   - opcjonalnie `ALITIGATOR_CREDIT_PACKS_JSON` dla starszych klientów korzystających z pakietów
   - `ALITIGATOR_CREDIT_COST_PER_QUERY`

Stripe webhook powinien wskazywać na:

- `POST /api/billing/webhooks/stripe`

Ręczne granty kredytów:

- źródłem prawdy o adminie jest `public.profiles.is_admin`
- `ALITIGATOR_ADMIN_EMAILS` można zostawić tymczasowo do zbootstrapowania istniejących adminów do `profiles.is_admin = true`
- admin może przyznać kredyty bez Stripe przez `POST /api/admin/credits/grant`
- w UI pojawia się wtedy prosty formularz do dopisywania kredytów po e-mailu użytkownika

Przyznanie admina bezpośrednio w Supabase:

```sql
update public.profiles
set is_admin = true
where email = 'stanislawwadolowski123@gmail.com';
```

### Lokalny webhook przez Stripe CLI

Jeśli nie masz jeszcze domeny, webhook do developmentu odpalaj lokalnie przez Stripe CLI.

Przepływ:

1. Uruchom backend lokalnie na `http://127.0.0.1:8000`
2. Zaloguj Stripe CLI do swojego konta
3. Forwarduj eventy do:
   - `http://127.0.0.1:8000/api/billing/webhooks/stripe`
4. Skopiuj wypisany przez CLI secret `whsec_...`
5. Wstaw go lokalnie do `apps/api/.env` jako:
   - `STRIPE_WEBHOOK_SECRET=whsec_...`

Przykładowa komenda:

```bash
stripe listen --forward-to http://127.0.0.1:8000/api/billing/webhooks/stripe
```

Do tego flow w naszym backendzie wystarczą eventy:

- `checkout.session.completed`
- `checkout.session.expired`
- `checkout.session.async_payment_failed`

Bezpieczeństwo na czas developmentu:

- do lokalnych testów używaj testowego klucza Stripe, nie `sk_live_...`
- secret z `stripe listen` jest tylko lokalny i nie będzie taki sam jak secret produkcyjnego webhooka po deploymencie

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
- opcjonalny: MariaDB/MySQL jako zewnętrzny magazyn corpusu.

Domyślny przepływ jest taki:

- importer dalej dopisuje nowe interpretacje do `processed` JSONL jako lokalnego bufora roboczego,
- `POST /api/rag/reindex` domyślnie buduje lokalny indeks SQLite,
- `/api/chat` korzysta z lokalnego indeksu, a Supabase jest tylko opcjonalnym rozszerzeniem.
- jeśli pliki źródłowe ustaw są nowsze niż lokalny SQLite, backend automatycznie odświeży indeks przy pierwszym zapytaniu.
- chat używa retrievalu chunkowego do wyboru dokumentów, a następnie domyślnie odtwarza pełną treść do 6 wybranych dokumentów i przekazuje ją modelowi do wewnętrznej selekcji oraz syntezy.

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

## MariaDB/MySQL dla corpusu RAG

Jeśli chcesz trzymać interpretacje i wyroki poza plikiem SQLite, backend obsługuje osobny storage RAG w MariaDB/MySQL.

1. Utwórz schemat z:
   - `apps/api/sql/rag_mysql_schema.sql`
2. Ustaw w `apps/api/.env`:
   - `ALITIGATOR_RAG_BACKEND=mysql`
   - `ALITIGATOR_RAG_MYSQL_HOST=...`
   - `ALITIGATOR_RAG_MYSQL_PORT=3306`
   - `ALITIGATOR_RAG_MYSQL_DATABASE=...`
   - `ALITIGATOR_RAG_MYSQL_USER=...`
   - `ALITIGATOR_RAG_MYSQL_PASSWORD=...`
   - opcjonalnie `ALITIGATOR_RAG_MYSQL_DOCUMENTS_TABLE=rag_documents`
   - opcjonalnie `ALITIGATOR_RAG_MYSQL_CHUNKS_TABLE=rag_chunks`
   - opcjonalnie `ALITIGATOR_RAG_MYSQL_SSL_DISABLED=true`, jeśli hosting nie wspiera SSL dla połączeń MySQL
3. Uruchom reindeksację:

```bash
curl -X POST http://127.0.0.1:8000/api/rag/reindex \
	-H 'Content-Type: application/json' \
	-d '{"force":true}'
```

Po przełączeniu backendu na `mysql`:

- `/api/chat` korzysta z MariaDB/MySQL jako źródła dokumentów RAG,
- `/api/rag/search` odpytuje MariaDB/MySQL,
- Supabase nadal obsługuje auth, profile, chat storage i billing.

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
- schemat kont i billingu też zakłada `public` oraz `auth.users` jako źródło tożsamości
- tabele bazowe: `profiles`, `credit_ledger`, `credit_orders`, `chat_threads`, `chat_messages`
- RLS jest włączone na tabelach użytkownika
- sekret `SUPABASE_SECRET_KEY` ma pozostać wyłącznie po stronie backendu
