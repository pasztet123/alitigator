from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

from app.supabase_rag import is_supabase_sync_enabled, sync_records_to_supabase

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_RAW_OUTPUT_PATH = DATA_DIR / "raw" / "eureka_interpretations.raw.jsonl"
DEFAULT_PROCESSED_OUTPUT_PATH = DATA_DIR / "processed" / "eureka_interpretations.jsonl"

API_BASE_URL = "https://eureka.mf.gov.pl/api/public/v1"
PUBLIC_DETAIL_URL = "https://eureka.mf.gov.pl/informacje/podglad/{document_id}"
SEARCH_ENDPOINT = "/wyszukiwarka/informacje/"
DETAIL_ENDPOINT_TEMPLATE = "/informacje/{document_id}"

DEFAULT_SORT = "DT_WYD,desc"
DEFAULT_PAGE_SIZE = 20
DEFAULT_CONCURRENCY = 1
DEFAULT_REQUEST_TIMEOUT = 45.0
DEFAULT_RETRY_COUNT = 3

SEARCH_COLUMNS = [
    "KATEGORIA_INFORMACJI",
    "SYG",
    "DT_WYD",
    "AUTOR",
    "ID_INFORMACJI",
    "TEZA",
    "SLOWA_KLUCZOWE",
    "PRZEPISY",
    "ZAGADNIENIA",
    "NOMENKLATURA_SCALONA",
    "KLASYFIKACJA_PKWIU",
    "KLASYFIKACJA_PKOB",
    "RODZAJ_WYROBU_AKCYZOWEGO",
    "STATUS_INFORMACJI",
    "DATA_REJESTRACJI",
    "KOMENTARZE_BIP",
    "KOM_BIP_OPIS",
    "NR_INTERP_ZAPYT",
    "SYGN_ODP",
    "DATA_ODP",
    "SYGNATURA_ORZECZENIA",
    "DATA_ORZECZENIA",
    "SADY",
    "DATA_WYSLANIA_NEWSLETTERA",
    "ID_INFORMACJI",
]

CONTENT_FIELD_KEYS = [
    "TRESC_INTERESARIUSZ",
    "TRESC_INTERPRETACJI",
    "TRESC",
    "UZASADNIENIE",
    "TRESC_HTML",
]

BLOCK_TAGS = {
    "article",
    "br",
    "div",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "li",
    "ol",
    "p",
    "section",
    "table",
    "tr",
    "ul",
}


class EurekaIngestError(RuntimeError):
    pass


class HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self.parts.append(f"&#{name};")


@dataclass
class FetchConfig:
    query: str = ""
    page_size: int = DEFAULT_PAGE_SIZE
    start_page: int = 0
    max_pages: int | None = None
    limit: int | None = 1000
    concurrency: int = DEFAULT_CONCURRENCY
    retry_count: int = DEFAULT_RETRY_COUNT
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    pause_seconds: float = 0.0
    category: str | None = "Interpretacja indywidualna"
    law_tags: list[str] | None = None
    published_dates: list[str] | None = None
    sort: str = DEFAULT_SORT
    raw_output_path: str | None = None
    output_path: str | None = None
    overwrite: bool = False

    @property
    def processed_output_path(self) -> Path:
        if self.output_path:
            return Path(self.output_path)
        return DEFAULT_PROCESSED_OUTPUT_PATH

    @property
    def resolved_raw_output_path(self) -> Path:
        if self.raw_output_path:
            return Path(self.raw_output_path)
        return DEFAULT_RAW_OUTPUT_PATH


def normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def html_to_text(html: str) -> str:
    if not html:
        return ""

    parser = HTMLToTextParser()
    parser.feed(html)
    parser.close()

    raw_text = unescape("".join(parser.parts))
    cleaned_lines: list[str] = []
    previous_blank = False

    for raw_line in raw_text.splitlines():
        line = normalize_spaces(raw_line)
        if not line:
            if not previous_blank:
                cleaned_lines.append("")
            previous_blank = True
            continue

        cleaned_lines.append(line)
        previous_blank = False

    return "\n".join(cleaned_lines).strip()


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue

            document_id = row.get("document_id") or row.get("source_id")
            if document_id is not None:
                seen_ids.add(str(document_id))

    return seen_ids


def write_jsonl_row(output_file: Any, row: dict[str, Any]) -> None:
    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def first_string(value: Any) -> str | None:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
        return None
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def extract_field_map(detail: dict[str, Any]) -> dict[str, Any]:
    field_map: dict[str, Any] = {}
    fields = ((detail.get("dokument") or {}).get("fields") or [])

    for field in fields:
        key = field.get("key")
        if not key:
            continue
        field_map[str(key)] = field.get("value")

    return field_map


def derive_law_tags(summary: dict[str, Any]) -> list[str]:
    tags: set[str] = set()

    for source in (summary.get("PRZEPISY") or []):
        if not isinstance(source, str):
            continue
        remainder = source
        while remainder.startswith("[") and "]" in remainder:
            tag, remainder = remainder[1:].split("]", 1)
            if tag:
                tags.add(f"[{tag}]")

    return sorted(tags)


def find_content_html(field_map: dict[str, Any]) -> str:
    for key in CONTENT_FIELD_KEYS:
        value = field_map.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_processed_row(summary: dict[str, Any], detail: dict[str, Any], *, query: str) -> dict[str, Any]:
    field_map = extract_field_map(detail)
    content_html = find_content_html(field_map)
    content_text = html_to_text(content_html)
    document_id = str(summary.get("ID_INFORMACJI") or field_map.get("ID_INFORMACJI") or detail.get("id"))

    attachments = ((detail.get("dokument") or {}).get("zalacznikiContent") or [])

    return {
        "source": "eureka",
        "source_type": "interpretation",
        "source_subtype": "general" if "ogóln" in str(summary.get("KATEGORIA_INFORMACJI") or "").lower() else "individual",
        "document_id": document_id,
        "index": document_id,
        "version_id": detail.get("versionId"),
        "template_id": detail.get("szablonId"),
        "template_version_id": detail.get("wersjaSzablonuId"),
        "category": first_string(summary.get("KATEGORIA_INFORMACJI")) or detail.get("nazwa"),
        "status": first_string(summary.get("STATUS_INFORMACJI")) or first_string(field_map.get("STATUS_INFORMACJI")),
        "subject": summary.get("TEZA") or field_map.get("TEZA") or detail.get("nazwa"),
        "signature": summary.get("SYG") or field_map.get("SYG"),
        "author": first_string(summary.get("AUTOR")) or first_string(field_map.get("AUTOR")),
        "published_date": field_map.get("DT_WYD") or summary.get("DT_WYD"),
        "published_at": field_map.get("DATA_PUBLIKACJI") or summary.get("DATA_REJESTRACJI"),
        "keywords": [str(value) for value in as_list(summary.get("SLOWA_KLUCZOWE")) if str(value).strip()],
        "legal_provisions": [str(value) for value in as_list(summary.get("PRZEPISY")) if str(value).strip()],
        "issues": [str(value) for value in as_list(summary.get("ZAGADNIENIA")) if str(value).strip()],
        "law_tags": derive_law_tags(summary),
        "query": query,
        "source_url": PUBLIC_DETAIL_URL.format(document_id=document_id),
        "content_html": content_html,
        "content_text": content_text,
        "content_sha256": hashlib.sha256(content_text.encode("utf-8")).hexdigest() if content_text else None,
        "attachments": attachments,
        "raw_field_map": field_map,
        "raw_search": summary,
        "raw_detail": detail,
        "retrieved_at": iso_now(),
    }


def matches_category(summary: dict[str, Any], category: str | None) -> bool:
    if category is None:
        return True
    categories = [str(value) for value in as_list(summary.get("KATEGORIA_INFORMACJI"))]
    return category in categories


def matches_law_tags(summary: dict[str, Any], law_tags: list[str]) -> bool:
    if not law_tags:
        return True

    haystack = [str(value) for value in as_list(summary.get("PRZEPISY")) + as_list(summary.get("ZAGADNIENIA"))]
    combined = "\n".join(haystack)
    return any(tag in combined for tag in law_tags)


async def request_json(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    retry_count: int,
    request_timeout: float,
    **kwargs: Any,
) -> Any:
    last_error: Exception | None = None

    for attempt in range(1, retry_count + 1):
        try:
            response = await asyncio.wait_for(
                client.request(method, url, **kwargs),
                timeout=max(1.0, request_timeout + 1.0),
            )
            response.raise_for_status()
            return response.json()
        except (asyncio.TimeoutError, TimeoutError, httpx.HTTPError, json.JSONDecodeError) as error:
            last_error = error
            if attempt >= retry_count:
                break
            await asyncio.sleep(min(5.0, attempt * 1.5))

    if last_error is None:
        raise EurekaIngestError(f"Request failed without an explicit error: {method} {url}")
    raise EurekaIngestError(f"{method} {url} failed: {last_error!r}") from last_error


async def search_page(client: httpx.AsyncClient, *, page_number: int, options: FetchConfig) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if options.published_dates:
        filters["DT_WYD"] = options.published_dates

    payload = {
        "filter": filters,
        "columns": SEARCH_COLUMNS,
        "searchInFullPhrase": True,
        "searchInContent": False,
        "searchInSynonyms": False,
        "searchQuery": options.query,
        "warunkiDodatkowe": [],
    }
    params = {
        "size": options.page_size,
        "page": page_number,
        "sort": options.sort,
    }
    return await request_json(
        client,
        "POST",
        SEARCH_ENDPOINT,
        retry_count=options.retry_count,
        request_timeout=options.request_timeout,
        params=params,
        json=payload,
    )


async def fetch_detail(client: httpx.AsyncClient, document_id: str, *, retry_count: int) -> dict[str, Any]:
    endpoint = DETAIL_ENDPOINT_TEMPLATE.format(document_id=document_id)
    return await request_json(
        client,
        "GET",
        endpoint,
        retry_count=retry_count,
        request_timeout=client.timeout.read if client.timeout.read is not None else DEFAULT_REQUEST_TIMEOUT,
    )


async def fetch_detail_batch(
    client: httpx.AsyncClient,
    summaries: list[dict[str, Any]],
    *,
    options: FetchConfig,
) -> list[tuple[dict[str, Any], dict[str, Any] | Exception]]:
    async def fetch_one(summary: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | Exception]:
        document_id = str(summary["ID_INFORMACJI"])
        try:
            detail = await fetch_detail(client, document_id, retry_count=options.retry_count)
            return summary, detail
        except Exception as error:  # noqa: BLE001
            return summary, error

    return await asyncio.gather(*(fetch_one(summary) for summary in summaries))


async def fetch_latest_interpretations(config: FetchConfig, *, progress_callback: Any = None) -> dict[str, Any]:
    raw_output_path = config.resolved_raw_output_path
    processed_output_path = config.processed_output_path
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_output_path.parent.mkdir(parents=True, exist_ok=True)

    if config.overwrite:
        if raw_output_path.exists():
            raw_output_path.unlink()
        if processed_output_path.exists():
            processed_output_path.unlink()

    seen_ids = set() if config.overwrite else read_existing_ids(processed_output_path)
    total_written = 0
    total_seen = len(seen_ids)
    total_hits: int | None = None
    failed_ids: list[str] = []
    last_document_id: str | None = None

    async with httpx.AsyncClient(
        base_url=API_BASE_URL,
        timeout=config.request_timeout,
        headers={"Accept": "application/json"},
    ) as client:
        with raw_output_path.open("a", encoding="utf-8") as raw_file, processed_output_path.open("a", encoding="utf-8") as processed_file:
            page_number = max(0, config.start_page)

            while True:
                if config.max_pages is not None and page_number >= config.max_pages:
                    break
                if config.limit is not None and total_written >= config.limit:
                    break

                response = await search_page(client, page_number=page_number, options=config)
                results = response.get("results") or []
                if total_hits is None:
                    total_hits = int(response.get("totalHits") or 0)

                if not results:
                    break

                filtered_summaries: list[dict[str, Any]] = []
                for summary in results:
                    document_id = str(summary.get("ID_INFORMACJI"))
                    if not document_id or document_id in seen_ids:
                        continue
                    if not matches_category(summary, config.category):
                        continue
                    if not matches_law_tags(summary, config.law_tags or []):
                        continue
                    filtered_summaries.append(summary)

                if progress_callback is not None:
                    progress_callback(
                        page=page_number,
                        fetched=len(results),
                        selected=len(filtered_summaries),
                        total_hits=total_hits,
                        written=total_written,
                    )

                if not filtered_summaries:
                    page_number += 1
                    if config.pause_seconds:
                        await asyncio.sleep(config.pause_seconds)
                    continue

                for batch_start in range(0, len(filtered_summaries), config.concurrency):
                    if config.limit is not None and total_written >= config.limit:
                        break

                    batch = filtered_summaries[batch_start : batch_start + config.concurrency]
                    pairs = await fetch_detail_batch(client, batch, options=config)
                    batch_processed_rows: list[dict[str, Any]] = []

                    for summary, detail_or_error in pairs:
                        document_id = str(summary["ID_INFORMACJI"])
                        if isinstance(detail_or_error, Exception):
                            failed_ids.append(document_id)
                            print(f"Failed detail fetch for {document_id}: {detail_or_error}", file=sys.stderr, flush=True)
                            continue

                        processed_row = build_processed_row(summary, detail_or_error, query=config.query)
                        raw_row = {
                            "document_id": document_id,
                            "query": config.query,
                            "retrieved_at": processed_row["retrieved_at"],
                            "summary": summary,
                            "detail": detail_or_error,
                        }

                        write_jsonl_row(raw_file, raw_row)
                        write_jsonl_row(processed_file, processed_row)
                        raw_file.flush()
                        processed_file.flush()
                        batch_processed_rows.append(processed_row)

                        seen_ids.add(document_id)
                        total_written += 1
                        total_seen += 1
                        last_document_id = document_id

                        if progress_callback is not None and total_written % 25 == 0:
                            progress_callback(page=page_number, saved=total_written, total_unique=total_seen)

                        if config.limit is not None and total_written >= config.limit:
                            break

                    if batch_processed_rows and is_supabase_sync_enabled():
                        try:
                            sync_records_to_supabase(batch_processed_rows, force=True)
                        except Exception as error:  # noqa: BLE001
                            raise EurekaIngestError(f"Supabase sync failed: {error}") from error

                page_number += 1
                if config.pause_seconds:
                    await asyncio.sleep(config.pause_seconds)

    return {
        "count": total_written,
        "output_path": str(processed_output_path),
        "raw_output_path": str(raw_output_path),
        "source": "eureka",
        "sort": config.sort,
        "last_document_id": last_document_id,
        "failed_ids": failed_ids,
        "total_unique_ids": total_seen,
    }


async def run_ingest(config: FetchConfig, *, progress_callback: Any = None) -> dict[str, Any]:
    result = await fetch_latest_interpretations(config, progress_callback=progress_callback)
    if config.limit is not None and result["count"] < config.limit:
        raise EurekaIngestError(
            f"Fetched only {result['count']} records, below requested minimum {config.limit}. "
            f"Processed file: {result['output_path']}"
        )
    return result


def parse_args() -> FetchConfig:
    parser = argparse.ArgumentParser(description="Download individual interpretations from the EUREKA public API.")
    parser.add_argument("--query", default="")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--retry-count", type=int, default=DEFAULT_RETRY_COUNT)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--category", default="Interpretacja indywidualna")
    parser.add_argument("--law-tag", action="append", dest="law_tags", default=[])
    parser.add_argument("--published-date", action="append", dest="published_dates", default=[])
    parser.add_argument("--sort", default=DEFAULT_SORT)
    parser.add_argument("--raw-output", default=str(DEFAULT_RAW_OUTPUT_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_PROCESSED_OUTPUT_PATH))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    category = args.category.strip() or None

    return FetchConfig(
        query=args.query,
        page_size=max(1, args.page_size),
        start_page=max(0, args.start_page),
        max_pages=args.max_pages,
        limit=args.limit,
        concurrency=max(1, args.concurrency),
        retry_count=max(1, args.retry_count),
        request_timeout=max(1.0, args.request_timeout),
        pause_seconds=max(0.0, args.pause_seconds),
        category=category,
        law_tags=[tag for tag in args.law_tags if tag],
        published_dates=[date for date in args.published_dates if date],
        sort=args.sort,
        raw_output_path=args.raw_output,
        output_path=args.output_path,
        overwrite=args.overwrite,
    )


def main() -> None:
    config = parse_args()

    def log_progress(**payload: Any) -> None:
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    result = asyncio.run(run_ingest(config, progress_callback=log_progress))
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
