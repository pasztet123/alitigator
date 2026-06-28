from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DEFAULT_RAW_OUTPUT_PATH = DATA_DIR / "raw" / "cbosa_nsa_fsk_judgments.raw.jsonl"
DEFAULT_PROCESSED_OUTPUT_PATH = DATA_DIR / "processed" / "cbosa_nsa_fsk_judgments.jsonl"

DEFAULT_BASE_URL = "https://orzeczenia.nsa.gov.pl"
DEFAULT_SEARCH_PATH = "/cbo/search"
DEFAULT_DETAIL_PATH_TEMPLATE = "/doc/{document_id}"

DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; alitigator-cbosa-ingest/1.0)"
DEFAULT_REQUEST_TIMEOUT = 45.0
DEFAULT_RETRY_COUNT = 3
DEFAULT_PAGE_SIZE = 100

DETAIL_LINK_RE = re.compile(r"/doc/([0-9A-Za-z_-]+)")
SIGNATURE_RE = re.compile(r"\b(?:I|II|III|IV|V|VI|VII|VIII)?\s*FSK\b[^<\n\r]{0,80}", re.IGNORECASE)
DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})[-.](\d{2})[-.](\d{2})\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")
CBOSA_LABELS = {
    "Data orzeczenia",
    "Data wpływu",
    "Sąd",
    "Sędziowie",
    "Symbol z opisem",
    "Hasła tematyczne",
    "Sygn. powiązane",
    "Skarżony organ",
    "Treść wyniku",
    "Powołane przepisy",
    "Sentencja",
    "Uzasadnienie",
}
CBOSA_STOP_LINES = {"Powrót do listy", "Powered by SoftProdukt"}
SYMBOL_DOMAIN_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("VAT", ("6110", "podatek od towarów i usług")),
    ("AKCYZA", ("6111", "podatek akcyzowy")),
    ("PIT", ("6112", "dochodowy od osób fizycznych")),
    ("CIT", ("6113", "dochodowy od osób prawnych")),
    ("SPADKI", ("6114", "spadków i darowizn")),
    ("NIERUCHOMOŚCI", ("6115", "podatki od nieruchomości", "podatek od nieruchomości")),
    ("PCC", ("6116", "czynności cywilnoprawnych", "opłata skarbowa")),
    ("ORDYNACJA", ("6117", "odpowiedzialność podatkowa", "ulgi płatnicze")),
    ("EGZEKUCJA", ("6118", "egzekucja świadczeń pieniężnych", "zabezpieczenie zobowiązań")),
)

SEARCH_PARAM_ALIASES: tuple[dict[str, str], ...] = (
    {
        "sygnatura": "FSK",
        "sad": "Naczelny Sąd Administracyjny",
        "rodzaj": "Wyrok",
        "submit": "Szukaj",
    },
)

BLOCK_TAGS = {
    "article",
    "br",
    "dd",
    "div",
    "dt",
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
    "td",
    "th",
    "tr",
    "ul",
}


class CbosaIngestError(RuntimeError):
    pass


class HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript"}:
            self.skip_depth += 1
            return
        if self.skip_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript"} and self.skip_depth:
            self.skip_depth -= 1
            return
        if self.skip_depth:
            return
        if tag in BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.skip_depth:
            self.parts.append(data)

    def handle_entityref(self, name: str) -> None:
        if not self.skip_depth:
            self.parts.append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        if not self.skip_depth:
            self.parts.append(f"&#{name};")


@dataclass(frozen=True)
class FetchConfig:
    base_url: str = DEFAULT_BASE_URL
    search_path: str = DEFAULT_SEARCH_PATH
    detail_path_template: str = DEFAULT_DETAIL_PATH_TEMPLATE
    signature_query: str = "FSK"
    court: str = "Naczelny Sąd Administracyjny"
    judgment_type: str = "Wyrok"
    start_page: int = 0
    max_pages: int | None = None
    limit: int | None = 1000
    page_size: int = DEFAULT_PAGE_SIZE
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT
    retry_count: int = DEFAULT_RETRY_COUNT
    search_params: str | None = None
    raw_output_path: str | None = None
    output_path: str | None = None
    overwrite: bool = False
    rebuild_from_raw: bool = False
    progress_every: int = 25
    pause_seconds: float = 0.5

    @property
    def processed_output_path(self) -> Path:
        return Path(self.output_path) if self.output_path else DEFAULT_PROCESSED_OUTPUT_PATH

    @property
    def resolved_raw_output_path(self) -> Path:
        return Path(self.raw_output_path) if self.raw_output_path else DEFAULT_RAW_OUTPUT_PATH


def normalize_spaces(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def html_to_text(html: str) -> str:
    parser = HTMLTextParser()
    parser.feed(html)
    parser.close()
    raw_text = unescape("".join(parser.parts))
    lines: list[str] = []
    previous_blank = False
    for raw_line in raw_text.splitlines():
        line = normalize_spaces(raw_line)
        if not line:
            if not previous_blank:
                lines.append("")
            previous_blank = True
            continue
        lines.append(line)
        previous_blank = False
    return "\n".join(lines).strip()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl_row(output_file: Any, row: dict[str, Any]) -> None:
    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            document_id = row.get("document_id") or row.get("source_id")
            if document_id:
                ids.add(str(document_id))
    return ids


def request_text(
    client: httpx.Client,
    method: str,
    url: str,
    *,
    retry_count: int,
    request_timeout: float,
    data: dict[str, str] | None = None,
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retry_count + 1):
        try:
            response = client.request(method, url, timeout=request_timeout, data=data)
            response.raise_for_status()
            return response.text
        except httpx.HTTPError as error:
            last_error = error
            if attempt >= retry_count:
                break
            time.sleep(min(10.0, attempt * 2.0))
    raise CbosaIngestError(f"{method} {url} failed: {last_error!r}")


def parse_search_params(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    parsed = parse_qs(value, keep_blank_values=True)
    if parsed:
        return {key: values[-1] for key, values in parsed.items() if values}
    params: dict[str, str] = {}
    for pair in value.split(","):
        if "=" not in pair:
            continue
        key, raw_param_value = pair.split("=", 1)
        params[key.strip()] = raw_param_value.strip()
    return params


def format_search_params(params: dict[str, str], options: FetchConfig, page: int) -> dict[str, str]:
    formatted: dict[str, str] = {}
    replacements = {
        "{page}": str(page),
        "{page_size}": str(options.page_size),
        "{signature}": options.signature_query,
        "{court}": options.court,
        "{judgment_type}": options.judgment_type,
    }
    for key, value in params.items():
        formatted_value = value
        for token, replacement in replacements.items():
            formatted_value = formatted_value.replace(token, replacement)
        formatted[key] = formatted_value
    return formatted


def build_search_requests(options: FetchConfig, page: int) -> list[tuple[str, dict[str, str]]]:
    custom_params = parse_search_params(options.search_params)
    param_sets = [custom_params] if custom_params else list(SEARCH_PARAM_ALIASES)
    requests: list[tuple[str, dict[str, str]]] = []
    base_search_url = urljoin(options.base_url.rstrip("/") + "/", options.search_path.lstrip("/"))
    for params in param_sets:
        formatted = format_search_params(params, options, page)
        requests.append((base_search_url, formatted))
    return requests


def build_search_urls(options: FetchConfig, page: int) -> list[str]:
    return [f"{url}?{urlencode(data)}" for url, data in build_search_requests(options, page)]


def extract_detail_ids(html: str) -> list[str]:
    seen: set[str] = set()
    ids: list[str] = []
    anchors = re.findall(r"<a\b[^>]*href=['\"](/doc/([0-9A-Za-z_-]+))['\"][^>]*>(.*?)</a>", html, flags=re.IGNORECASE | re.DOTALL)
    if anchors:
        candidates = [document_id for _, document_id, label_html in anchors if "FSK" in strip_tags(label_html).upper()]
    else:
        candidates = [match.group(1) for match in DETAIL_LINK_RE.finditer(html)]

    for document_id in candidates:
        if document_id in seen:
            continue
        seen.add(document_id)
        ids.append(document_id)
    return ids


def strip_tags(fragment: str) -> str:
    return normalize_spaces(unescape(HTML_TAG_RE.sub(" ", fragment)))


def extract_title(html: str) -> str:
    for pattern in (r"<h1[^>]*>(.*?)</h1>", r"<title[^>]*>(.*?)</title>"):
        match = re.search(pattern, html, re.IGNORECASE | re.DOTALL)
        if match:
            return strip_tags(match.group(1))
    return "Wyrok NSA - Izba Finansowa"


def extract_signature(html: str, fallback: str = "") -> str:
    text = html_to_text(html)
    for line in text.splitlines():
        if "FSK" not in line.upper():
            continue
        match = SIGNATURE_RE.search(line)
        if match:
            return normalize_spaces(match.group(0))
        line = normalize_spaces(line)
        if line:
            return line[:120]
    return fallback


def extract_judgment_date(html: str) -> str | None:
    text = html_to_text(html)
    for line in text.splitlines():
        lowered = line.lower()
        if "data" not in lowered and "orzeczenia" not in lowered and "wydania" not in lowered:
            continue
        match = DATE_RE.search(line)
        if match:
            year, month, day = match.groups()
            return f"{year}-{month}-{day}"
    match = DATE_RE.search(text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def extract_keywords(html: str) -> list[str]:
    text = html_to_text(html)
    keywords: set[str] = set()
    for line in text.splitlines():
        lowered = line.lower()
        if lowered in {"symbol z opisem", "hasła tematyczne", "słowa kluczowe", "slowa kluczowe"}:
            continue
        if any(marker in lowered for marker in ("hasła tematyczne", "symbol", "słowa kluczowe", "slowa kluczowe")):
            cleaned = re.sub(r"^[^:：]{0,40}[:：]", "", line).strip()
            for item in re.split(r"[,;|]", cleaned):
                value = normalize_spaces(item)
                if value and len(value) <= 80:
                    keywords.add(value)
    return sorted(keywords)


def extract_cbosa_fields(text: str) -> dict[str, str]:
    fields: dict[str, list[str]] = {}
    current_label: str | None = None
    for raw_line in text.splitlines():
        line = normalize_spaces(raw_line)
        if not line:
            continue
        if line in CBOSA_STOP_LINES:
            current_label = None
            continue
        if line in CBOSA_LABELS:
            current_label = line
            fields.setdefault(line, [])
            continue
        if current_label:
            fields[current_label].append(line)
    return {label: normalize_spaces(" ".join(parts)) for label, parts in fields.items() if parts}


def split_field_values(value: str) -> list[str]:
    values: list[str] = []
    for item in re.split(r"\s*[;,|]\s*", value or ""):
        normalized = normalize_spaces(item)
        if normalized and normalized not in values:
            values.append(normalized)
    return values


def extract_first_date(value: str) -> str | None:
    match = DATE_RE.search(value or "")
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{month}-{day}"


def derive_symbol_domains(symbol_text: str) -> list[str]:
    normalized = symbol_text.lower()
    domains: list[str] = []
    for domain, markers in SYMBOL_DOMAIN_MARKERS:
        if any(marker.lower() in normalized for marker in markers):
            domains.append(domain)
    return domains


def extract_cbosa_legal_provisions(text: str, fields: dict[str, str]) -> list[str]:
    provisions = set(extract_legal_provisions(text))
    cited = fields.get("Powołane przepisy", "")
    if cited:
        for value in re.split(r"(?=Dz\.U\.)|\s{2,}|;", cited):
            normalized = normalize_spaces(value)
            if normalized and len(normalized) <= 220:
                provisions.add(normalized)
    return sorted(provisions)


def build_cbosa_content_text(text: str, fields: dict[str, str], *, signature: str, subject: str) -> str:
    parts: list[str] = []
    if subject:
        parts.append(subject)
    for label in (
        "Symbol z opisem",
        "Hasła tematyczne",
        "Skarżony organ",
        "Treść wyniku",
        "Powołane przepisy",
        "Sygn. powiązane",
    ):
        value = fields.get(label)
        if value:
            parts.append(f"{label}: {value}")
    if signature:
        parts.append(f"Sygnatura: {signature}")
    for label in ("Sentencja", "Uzasadnienie"):
        value = fields.get(label)
        if value:
            parts.append(f"{label}\n{value}")
    if text:
        parts.append(f"Pełna treść\n{text}")
    return "\n\n".join(part for part in parts if part).strip()


def extract_legal_provisions(text: str) -> list[str]:
    provisions = set(re.findall(r"\bart\.\s*\d+[a-z]*(?:\s*[§ustpktlit.\w]+){0,8}", text, flags=re.IGNORECASE))
    return sorted(normalize_spaces(value) for value in provisions if len(value) <= 120)


def source_url_for_detail(options: FetchConfig, document_id: str) -> str:
    path = options.detail_path_template.format(document_id=document_id)
    return urljoin(options.base_url.rstrip("/") + "/", path.lstrip("/"))


def build_processed_row(document_id: str, detail_html: str, *, options: FetchConfig) -> dict[str, Any]:
    plain_text = html_to_text(detail_html)
    fields = extract_cbosa_fields(plain_text)
    signature = extract_signature(detail_html)
    judgment_date = extract_first_date(fields.get("Data orzeczenia", "")) or extract_judgment_date(detail_html)
    title = extract_title(detail_html)
    subject = title
    if signature and signature not in subject:
        subject = f"{signature} - {title}"
    symbol_text = fields.get("Symbol z opisem", "")
    thematic_text = fields.get("Hasła tematyczne", "")
    result_text = fields.get("Treść wyniku", "")
    authority = fields.get("Sąd") or "Naczelny Sąd Administracyjny"
    appealed_authority = fields.get("Skarżony organ", "")
    related_signatures = split_field_values(fields.get("Sygn. powiązane", ""))
    symbol_values = [value for value in (symbol_text, thematic_text) if value]
    symbol_domains = derive_symbol_domains(symbol_text)
    content_text = build_cbosa_content_text(plain_text, fields, signature=signature, subject=subject)
    legal_provisions = extract_cbosa_legal_provisions(content_text, fields)
    keywords = sorted({*extract_keywords(detail_html), *symbol_values, *symbol_domains})
    issues = ["Izba Finansowa", "NSA", "FSK"]
    for value in [*symbol_values, result_text, appealed_authority, *related_signatures]:
        if value and value not in issues:
            issues.append(value)
    law_tags = [f"[{domain}]" for domain in symbol_domains]

    return {
        "source": "cbosa",
        "source_type": "judgment",
        "source_subtype": "nsa",
        "document_id": f"cbosa:{document_id}",
        "index": document_id,
        "category": normalize_spaces(" - ".join(part for part in ["Wyrok NSA - Izba Finansowa", symbol_text] if part)),
        "subject": subject,
        "signature": signature or None,
        "author": authority,
        "authority": authority,
        "jurisdiction": "PL",
        "published_date": judgment_date,
        "judgment_date": judgment_date,
        "receipt_date": fields.get("Data wpływu") or None,
        "appealed_authority": appealed_authority or None,
        "result": result_text or None,
        "symbol_description": symbol_text or None,
        "related_signatures": related_signatures,
        "sentence_text": fields.get("Sentencja") or "",
        "justification_text": fields.get("Uzasadnienie") or "",
        "keywords": keywords,
        "legal_provisions": legal_provisions,
        "issues": issues,
        "law_tags": law_tags,
        "source_url": source_url_for_detail(options, document_id),
        "content_html": detail_html,
        "content_text": content_text,
        "content_sha256": hashlib.sha256(content_text.encode("utf-8")).hexdigest() if content_text else None,
        "raw_detail": {"document_id": document_id, "html": detail_html},
        "retrieved_at": iso_now(),
    }


def rebuild_processed_from_raw(options: FetchConfig) -> dict[str, Any]:
    raw_path = options.resolved_raw_output_path
    output_path = options.processed_output_path
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw CBOSA JSONL not found: {raw_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids: set[str] = set()
    processed = 0
    skipped_duplicate = 0
    skipped_non_fsk = 0
    with raw_path.open("r", encoding="utf-8") as raw_file, output_path.open("w", encoding="utf-8") as output_file:
        for line in raw_file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("type") != "detail":
                continue
            detail_id = str(row.get("document_id") or "").strip()
            detail_html = str(row.get("html") or "")
            document_id = f"cbosa:{detail_id}"
            if not detail_id or not detail_html:
                continue
            if document_id in seen_ids:
                skipped_duplicate += 1
                continue
            processed_row = build_processed_row(detail_id, detail_html, options=options)
            if not is_target_fsk_judgment(processed_row):
                skipped_non_fsk += 1
                continue
            write_jsonl_row(output_file, processed_row)
            seen_ids.add(document_id)
            processed += 1

    return {
        "processed": processed,
        "skipped_duplicate": skipped_duplicate,
        "skipped_non_fsk": skipped_non_fsk,
        "output_path": str(output_path),
        "raw_output_path": str(raw_path),
    }


def is_target_fsk_judgment(row: dict[str, Any]) -> bool:
    signature = str(row.get("signature") or "")
    text = f"{signature}\n{row.get('subject') or ''}\n{row.get('content_text') or ''}"
    return "FSK" in text.upper()


def fetch_search_page(client: httpx.Client, options: FetchConfig, page: int) -> tuple[str, list[str], str]:
    errors: list[str] = []
    if page > 0:
        find_url = urljoin(options.base_url.rstrip("/") + "/", f"cbo/find?p={page + 1}")
        html = request_text(client, "GET", find_url, retry_count=options.retry_count, request_timeout=options.request_timeout)
        ids = extract_detail_ids(html)
        if ids:
            return find_url, ids, html
        raise CbosaIngestError(f"No FSK /doc/ links found at {find_url}")

    for url, data in build_search_requests(options, page):
        try:
            html = request_text(client, "POST", url, retry_count=options.retry_count, request_timeout=options.request_timeout, data=data)
        except CbosaIngestError as error:
            errors.append(str(error))
            continue
        ids = extract_detail_ids(html)
        if ids:
            return f"{url}?{urlencode(data)}", ids, html
        errors.append(f"No FSK /doc/ links found at {url}?{urlencode(data)}")
    raise CbosaIngestError("; ".join(errors))


def run_ingest(options: FetchConfig) -> dict[str, Any]:
    output_path = options.processed_output_path
    raw_output_path = options.resolved_raw_output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw_output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_ids = set() if options.overwrite else read_existing_ids(output_path)
    mode = "w" if options.overwrite else "a"
    fetched = 0
    skipped_existing = 0
    skipped_non_fsk = 0
    failed_details = 0
    pages_processed = 0
    detail_ids_seen: set[str] = set()

    headers = {"Accept": "text/html,application/xhtml+xml", "User-Agent": DEFAULT_USER_AGENT}
    with httpx.Client(headers=headers, follow_redirects=True, timeout=options.request_timeout) as client:
        with output_path.open(mode, encoding="utf-8") as output_file, raw_output_path.open(mode, encoding="utf-8") as raw_file:
            if options.start_page > 0:
                fetch_search_page(client, options, 0)
            page = options.start_page
            while True:
                if options.max_pages is not None and pages_processed >= options.max_pages:
                    break
                if options.limit is not None and fetched >= options.limit:
                    break

                search_url, detail_ids, search_html = fetch_search_page(client, options, page)
                pages_processed += 1
                if not detail_ids:
                    break

                write_jsonl_row(raw_file, {"type": "search", "page": page, "url": search_url, "html": search_html, "retrieved_at": iso_now()})
                new_on_page = 0
                for detail_id in detail_ids:
                    if detail_id in detail_ids_seen:
                        continue
                    detail_ids_seen.add(detail_id)
                    document_id = f"cbosa:{detail_id}"
                    if document_id in seen_ids:
                        skipped_existing += 1
                        continue

                    detail_url = source_url_for_detail(options, detail_id)
                    try:
                        detail_html = request_text(client, "GET", detail_url, retry_count=options.retry_count, request_timeout=options.request_timeout)
                    except CbosaIngestError as error:
                        failed_details += 1
                        write_jsonl_row(
                            raw_file,
                            {
                                "type": "detail_error",
                                "document_id": detail_id,
                                "url": detail_url,
                                "error": str(error),
                                "retrieved_at": iso_now(),
                            },
                        )
                        raw_file.flush()
                        continue
                    write_jsonl_row(raw_file, {"type": "detail", "document_id": detail_id, "url": detail_url, "html": detail_html, "retrieved_at": iso_now()})
                    row = build_processed_row(detail_id, detail_html, options=options)
                    if not is_target_fsk_judgment(row):
                        skipped_non_fsk += 1
                        continue
                    write_jsonl_row(output_file, row)
                    output_file.flush()
                    raw_file.flush()
                    seen_ids.add(document_id)
                    fetched += 1
                    new_on_page += 1
                    if options.pause_seconds > 0:
                        time.sleep(options.pause_seconds)
                    if options.progress_every > 0 and fetched % options.progress_every == 0:
                        print(
                            json.dumps(
                                {
                                    "progress": True,
                                    "fetched": fetched,
                                    "page": page,
                                    "skipped_existing": skipped_existing,
                                    "skipped_non_fsk": skipped_non_fsk,
                                    "failed_details": failed_details,
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    if options.limit is not None and fetched >= options.limit:
                        break

                if options.progress_every > 0:
                    print(
                        json.dumps(
                            {
                                "page_done": page,
                                "page_results": len(detail_ids),
                                "new_on_page": new_on_page,
                                "fetched": fetched,
                                "skipped_existing": skipped_existing,
                                "failed_details": failed_details,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                page += 1

    return {
        "fetched": fetched,
        "skipped_existing": skipped_existing,
        "skipped_non_fsk": skipped_non_fsk,
        "failed_details": failed_details,
        "pages_processed": pages_processed,
        "output_path": str(output_path),
        "raw_output_path": str(raw_output_path),
    }


def parse_args(argv: list[str]) -> FetchConfig:
    parser = argparse.ArgumentParser(description="Import NSA Financial Chamber judgments from CBOSA into RAG-compatible JSONL.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--search-path", default=DEFAULT_SEARCH_PATH)
    parser.add_argument("--detail-path-template", default=DEFAULT_DETAIL_PATH_TEMPLATE)
    parser.add_argument("--signature-query", default="FSK")
    parser.add_argument("--court", default="Naczelny Sąd Administracyjny")
    parser.add_argument("--judgment-type", default="Wyrok")
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--retry-count", type=int, default=DEFAULT_RETRY_COUNT)
    parser.add_argument("--search-params", help="Override query params, e.g. 'sygnatura={signature}&sad=NSA&page={page}&pp={page_size}'.")
    parser.add_argument("--raw-output-path")
    parser.add_argument("--output-path")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-from-raw", action="store_true", help="Rebuild processed JSONL from existing raw detail rows without fetching CBOSA.")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--pause-seconds", type=float, default=0.5)
    args = parser.parse_args(argv)
    return FetchConfig(**vars(args))


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv or sys.argv[1:])
    result = rebuild_processed_from_raw(options) if getattr(options, "rebuild_from_raw", False) else run_ingest(options)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())