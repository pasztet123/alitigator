"""Extract and chunk core Polish tax treaties from locally downloaded PDFs.

The output mirrors the statute JSONL format used by the existing RAG pipeline,
but marks records as ``source_subtype=tax_treaty`` so retrieval can prefer
them for cross-border tax questions.

When an official MF PDF has no usable text layer, the chunker can fall back to
OCR and caches page-level results so rebuilds stay practical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pypdf import PdfReader
from PIL import Image

try:
    import fitz
    from rapidocr_onnxruntime import RapidOCR
except ModuleNotFoundError:  # pragma: no cover - optional runtime dependency
    fitz = None
    RapidOCR = None

REPO_ROOT = Path(__file__).resolve().parents[3]
OCR_CACHE_DIR = REPO_ROOT / "apps/api/data/laws/ocr_cache"
OCR_RENDER_SCALE = 1.5
OCR_ENGINE: Any | None = None

ARTICLE_RE = re.compile(r"Artyku[lł∏]\s+(\d+[A-Za-z]?)\b", re.IGNORECASE)
POLISH_KEEP_RE = re.compile(
    r"(artyku|umow|konwencj|podatk|dochod|maj[ąa]tk|zakres|zak[łl]ad|"
    r"dywidend|odset|nale[żz]no|zyski przedsi|miejsce zamieszkania|"
    r"siedzib|w[ał]a[śs]ciw|umawiaj|pa[nń]stw|nierezyd|opodatkow)",
    re.IGNORECASE,
)
FOREIGN_DROP_RE = re.compile(
    r"^(Article|Artikel|Agreement|Abkommen|Persons covered|Taxes covered|"
    r"This Agreement|The Agreement|Dieses Abkommen|Unter das Abkommen|"
    r"Allgemeine Begriffsbestimmungen|General definitions)\b",
    re.IGNORECASE,
)
HEADER_RE = re.compile(
    r"^(?:Dziennik Ustaw.*Poz\.\s*\d+|©Kancelaria Sejmu.*|[-–—]+\s*\d+\s*[-–—]+.*|"
    r"\d{4}-\d{2}-\d{2})\s*$",
    re.IGNORECASE,
)
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANKS_RE = re.compile(r"\n{3,}")
OCR_ARTICLE_WITH_NUMBER_RE = re.compile(r"\bArtyku[tłlI1|f]{1,3}\s*(\d+[A-Za-z]?)\b", re.IGNORECASE)
OCR_ARTICLE_FIX_RE = re.compile(r"\bArtyku[tłlI1|f]{1,3}\b", re.IGNORECASE)
OCR_ZAKLAD_FIX_RE = re.compile(r"\bZak[tlI1|][a4][dtlI1|]\b", re.IGNORECASE)


@dataclass(frozen=True)
class TreatySource:
    country: str
    slug: str
    variant: str
    pdf_path: Path
    source_url: str
    act_title: str
    subject_prefix: str
    publication: str
    legal_state_date: str
    published_date: str
    ready_without_ocr: bool = True
    structured_json_path: Path | None = None


CORE_TREATY_SOURCES: tuple[TreatySource, ...] = (
    TreatySource(
        country="Austria",
        slug="austria",
        variant="umowa",
        pdf_path=Path("resources/upo/austria/umowa_pl_en_de.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/21rjsgzo/d20051921_austria_umowa_pl_ang_niem.pdf",
        act_title="Umowa między Rzecząpospolitą Polską a Republiką Austrii w sprawie unikania podwójnego opodatkowania w zakresie podatków od dochodu i od majątku",
        subject_prefix="UPO Polska - Austria",
        publication="Dz.U. 2005 poz. 1921",
        legal_state_date="2005-10-26",
        published_date="2005-10-26",
    ),
    TreatySource(
        country="Czechy",
        slug="czechy",
        variant="umowa_2011",
        pdf_path=Path("resources/upo/czechy/umowa_2011_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/fu4nb3j0/czechy-nowa-umowa-pl.pdf",
        act_title="Umowa między Rzecząpospolitą Polską a Republiką Czeską w sprawie unikania podwójnego opodatkowania w zakresie podatków od dochodu",
        subject_prefix="UPO Polska - Czechy",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Francja",
        slug="francja",
        variant="umowa",
        pdf_path=Path("resources/upo/francja/umowa_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/w2bhjz2p/19750620_francja_konwencja_tekst_pl_e.pdf",
        act_title="Umowa między Rządem Polskiej Rzeczypospolitej Ludowej a Rządem Republiki Francuskiej w sprawie zapobieżenia podwójnemu opodatkowaniu w zakresie podatków od dochodu i majątku",
        subject_prefix="UPO Polska - Francja",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
    ),
    TreatySource(
        country="Irlandia",
        slug="irlandia",
        variant="umowa",
        pdf_path=Path("resources/upo/irlandia/umowa_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/ln5hlbnt/irlandia-konwencja-tekst-polski.pdf",
        act_title="Konwencja między Rzecząpospolitą Polską a Irlandią w sprawie unikania podwójnego opodatkowania",
        subject_prefix="UPO Polska - Irlandia",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Luksemburg",
        slug="luksemburg",
        variant="umowa",
        pdf_path=Path("resources/upo/luksemburg/umowa_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/ajkgmyrq/19950614_luksemburg_konwencja_tekst_pl_e.pdf",
        act_title="Konwencja między Rzecząpospolitą Polską a Wielkim Księstwem Luksemburga w sprawie unikania podwójnego opodatkowania w zakresie podatków od dochodu i majątku",
        subject_prefix="UPO Polska - Luksemburg",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
    ),
    TreatySource(
        country="Luksemburg",
        slug="luksemburg",
        variant="protokol_2012",
        pdf_path=Path("resources/upo/luksemburg/protokol_2012_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/3pzmp13g/20120607_luksemburg_tekst_pl_e.pdf",
        act_title="Protokół zmieniający Konwencję między Rzecząpospolitą Polską a Wielkim Księstwem Luksemburga",
        subject_prefix="UPO Polska - Luksemburg - protokół",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
    ),
    TreatySource(
        country="Niderlandy",
        slug="niderlandy",
        variant="umowa",
        pdf_path=Path("resources/upo/niderlandy/umowa_pl_en.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/vmppgq1g/holandia-konwencja-tekst-polski-angielski.pdf",
        act_title="Konwencja między Rzecząpospolitą Polską a Królestwem Niderlandów w sprawie unikania podwójnego opodatkowania",
        subject_prefix="UPO Polska - Niderlandy",
        publication="Dz.U. 2003 poz. 2120",
        legal_state_date="2003-10-09",
        published_date="2003-10-09",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Niderlandy",
        slug="niderlandy",
        variant="protokol_2022",
        pdf_path=Path("resources/upo/niderlandy/protokol_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/kr4pjone/protok%C3%B3%C5%82-tekst-polski.pdf",
        act_title="Protokół między Rzecząpospolitą Polską a Królestwem Niderlandów o zmianie Konwencji",
        subject_prefix="UPO Polska - Niderlandy - protokół",
        publication="Dz.U. 2022 poz. 906",
        legal_state_date="2022-04-28",
        published_date="2022-04-28",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Niemcy",
        slug="niemcy",
        variant="umowa",
        pdf_path=Path("resources/upo/niemcy/umowa_pl_de.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/wdxllckt/niemcy-konwencja-tekst-polski-niemiecki.pdf",
        act_title="Umowa między Rzecząpospolitą Polską a Republiką Federalną Niemiec w sprawie unikania podwójnego opodatkowania",
        subject_prefix="UPO Polska - Niemcy",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Szwajcaria",
        slug="szwajcaria",
        variant="umowa",
        pdf_path=Path("resources/upo/szwajcaria/umowa_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/fbol1ik4/19910902_konwencja_szwajcaria_pl_e.pdf",
        act_title="Konwencja między Rzecząpospolitą Polską a Konfederacją Szwajcarską w sprawie unikania podwójnego opodatkowania w zakresie podatków od dochodu i majątku",
        subject_prefix="UPO Polska - Szwajcaria",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
    ),
    TreatySource(
        country="Szwajcaria",
        slug="szwajcaria",
        variant="protokol_2010",
        pdf_path=Path("resources/upo/szwajcaria/protokol_2010_pl_en_de.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/3qlc1i1f/20100420_szwajcaria_protokol_obwieszczenie_tekst_pl_niem_ang.pdf",
        act_title="Protokół zmieniający Konwencję między Rzecząpospolitą Polską a Konfederacją Szwajcarską",
        subject_prefix="UPO Polska - Szwajcaria - protokół",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
    ),
    TreatySource(
        country="USA",
        slug="usa",
        variant="umowa_1974",
        pdf_path=Path("resources/upo/usa/umowa_1974_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/fygbbhi0/19741008_usa_konwencja_tekst_polski.pdf",
        act_title="Umowa między Rządem Polskiej Rzeczypospolitej Ludowej a Rządem Stanów Zjednoczonych Ameryki o uniknięciu podwójnego opodatkowania",
        subject_prefix="UPO Polska - USA",
        publication="Dz.U. 1976 poz. 178",
        legal_state_date="1976-07-30",
        published_date="1976-07-30",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Wielka Brytania",
        slug="wielka_brytania",
        variant="umowa",
        pdf_path=Path("resources/upo/wielka_brytania/umowa_pl.pdf"),
        structured_json_path=None,
        source_url="https://www.podatki.gov.pl/media/uanpfvts/wlk-brytania-konwencja-tekst-polski.pdf",
        act_title="Konwencja między Rzecząpospolitą Polską a Zjednoczonym Królestwem w sprawie unikania podwójnego opodatkowania",
        subject_prefix="UPO Polska - Wielka Brytania",
        publication="Dz.U. 2006 poz. 1840",
        legal_state_date="2006-12-08",
        published_date="2006-12-08",
        ready_without_ocr=False,
    ),
    TreatySource(
        country="Wielka Brytania",
        slug="wielka_brytania",
        variant="tekst_syntetyczny_mli",
        pdf_path=Path("resources/upo/wielka_brytania/tekst_syntetyczny_mli_pl.pdf"),
        structured_json_path=Path("resources/upo/wielka_brytania/upo_polska_wielka_brytania_mli_pl.json"),
        source_url="https://www.podatki.gov.pl/media/dmklglvr/upo-pl-uk-mli-tekst-syntetyczny-pl.pdf",
        act_title="Tekst syntetyczny Konwencji MLI oraz Konwencji między Rzecząpospolitą Polską a Zjednoczonym Królestwem",
        subject_prefix="UPO Polska - Wielka Brytania - tekst syntetyczny MLI",
        publication="MF treaty PDF",
        legal_state_date="",
        published_date="",
    ),
)


def normalize_text(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ").replace("\uf0b7", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = HEADER_RE.sub("", text)
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines())
    return BLANKS_RE.sub("\n\n", text).strip()


def normalize_ocr_line(text: str) -> str:
    text = normalize_text(text)
    text = OCR_ARTICLE_WITH_NUMBER_RE.sub(r"Artykuł \1", text)
    text = OCR_ARTICLE_FIX_RE.sub("Artykuł", text)
    text = OCR_ZAKLAD_FIX_RE.sub("Zakład", text)
    text = re.sub(r"\bRzeczapospolit[ae]\b", "Rzeczpospolita", text, flags=re.IGNORECASE)
    text = re.sub(r"\bopodatkowanl[ae]\b", "opodatkowania", text, flags=re.IGNORECASE)
    return text


def keep_polish_line(line: str) -> bool:
    normalized = line.strip()
    if not normalized:
        return False
    if HEADER_RE.match(normalized):
        return False
    if FOREIGN_DROP_RE.match(normalized) and not POLISH_KEEP_RE.search(normalized):
        return False
    polish_hits = len(POLISH_KEEP_RE.findall(normalized))
    foreign_hits = len(re.findall(r"\b(the|this|agreement|article|taxes|persons|unter|abkommen|artikel|dieses|vertrag|tax)\b", normalized, re.IGNORECASE))
    if polish_hits > foreign_hits:
        return True
    if ARTICLE_RE.search(normalized):
        return True
    return polish_hits > 0 and foreign_hits == 0


def build_page(number: int, lines: list[str], raw_chars: int) -> dict[str, Any]:
    filtered_lines = [line for line in lines if keep_polish_line(line)]
    return {
        "number": number,
        "raw_chars": raw_chars,
        "text": "\n".join(filtered_lines).strip(),
    }


def extract_pdf_text_pages(source: TreatySource) -> list[dict[str, Any]]:
    pdf_path = source.pdf_path if source.pdf_path.is_absolute() else REPO_ROOT / source.pdf_path
    reader = PdfReader(str(pdf_path))
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(reader.pages, start=1):
        raw = normalize_text(page.extract_text() or "")
        pages.append(build_page(number=index, lines=raw.splitlines(), raw_chars=len(raw)))
    return pages


def build_ocr_cache_path(source: TreatySource) -> Path:
    return OCR_CACHE_DIR / f"{source.slug}__{source.variant}.json"


def read_cached_ocr_pages(source: TreatySource) -> list[dict[str, Any]] | None:
    cache_path = build_ocr_cache_path(source)
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if payload.get("pdf_path") != str(source.pdf_path):
        return None
    return payload.get("pages")


def write_cached_ocr_pages(source: TreatySource, pages: list[dict[str, Any]]) -> None:
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = build_ocr_cache_path(source)
    payload = {
        "pdf_path": str(source.pdf_path),
        "source_url": source.source_url,
        "pages": pages,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def render_page_to_array(page: Any) -> np.ndarray:
    if fitz is None:
        raise RuntimeError("PyMuPDF is unavailable")
    pix = page.get_pixmap(matrix=fitz.Matrix(OCR_RENDER_SCALE, OCR_RENDER_SCALE), alpha=False)
    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return np.array(image)


def ocr_pages(source: TreatySource) -> list[dict[str, Any]]:
    cached = read_cached_ocr_pages(source)
    if cached:
        return cached
    if fitz is None or RapidOCR is None:
        return []
    pdf_path = source.pdf_path if source.pdf_path.is_absolute() else REPO_ROOT / source.pdf_path
    doc = fitz.open(pdf_path)
    global OCR_ENGINE
    if OCR_ENGINE is None:
        OCR_ENGINE = RapidOCR()
    pages: list[dict[str, Any]] = []
    for index, page in enumerate(doc, start=1):
        image = render_page_to_array(page)
        result, _ = OCR_ENGINE(image)
        lines = [normalize_ocr_line(item[1]) for item in (result or [])]
        pages.append(build_page(number=index, lines=lines, raw_chars=sum(len(line) for line in lines)))
    write_cached_ocr_pages(source, pages)
    return pages


def iter_article_records(source: TreatySource, pages: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    article_lines: list[str] = []
    article_pages: list[int] = []
    article_number: str | None = None

    def flush() -> Iterator[dict[str, Any]]:
        nonlocal article_lines, article_pages, article_number
        if article_number and article_lines:
            text = "\n".join(article_lines).strip()
            if text:
                yield {
                    "article": article_number,
                    "pages": sorted(set(article_pages)),
                    "text": text,
                }
        article_lines = []
        article_pages = []
        article_number = None

    for page in pages:
        if not page["text"]:
            continue
        for line in page["text"].splitlines():
            match = ARTICLE_RE.search(line)
            if match:
                yield from flush()
                article_number = match.group(1).lower()
                article_lines = [line[match.start() :].strip()]
                article_pages = [page["number"]]
                continue
            if article_lines:
                article_lines.append(line)
                article_pages.append(page["number"])
    yield from flush()


def build_record(source: TreatySource, article: dict[str, Any]) -> dict[str, Any]:
    content = article["text"]
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    legal_provision = f"art. {article['article']}"
    keywords = [
        source.country.lower(),
        "umowa o unikaniu podwójnego opodatkowania",
        "upo",
        "tax treaty",
        "zakład",
        "zyski przedsiębiorstw",
        "dywidendy",
        "odsetki",
        "należności licencyjne",
    ]
    law_tags = [
        "UPO",
        "TAX_TREATY",
        source.country.upper(),
        source.publication,
        source.variant.upper(),
    ]
    return {
        "document_id": f"pl-upo-{source.slug}-{source.variant}-{legal_provision.replace(' ', '-')}",
        "source": "mf",
        "source_type": "statute",
        "source_subtype": "tax_treaty",
        "authority": "Ministerstwo Finansów / umowa międzynarodowa",
        "jurisdiction": "PL",
        "act_title": source.act_title,
        "publication": source.publication,
        "legal_state_date": source.legal_state_date,
        "published_date": source.published_date,
        "subject": f"{source.subject_prefix} - {legal_provision}",
        "legal_provisions": [legal_provision],
        "keywords": keywords,
        "issues": ["cit", "pit", "wht", "upo"],
        "law_tags": law_tags,
        "source_url": source.source_url,
        "source_pdf": str(source.pdf_path),
        "source_pages": article["pages"],
        "pre_chunked": True,
        "content_text": content,
        "content_sha256": digest,
    }


def load_structured_json_records(source: TreatySource) -> list[dict[str, Any]]:
    if source.structured_json_path is None:
        return []
    json_path = source.structured_json_path if source.structured_json_path.is_absolute() else REPO_ROOT / source.structured_json_path
    if not json_path.exists():
        return []
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for article in payload.get("articles", []):
        article_number = str(article.get("article_number", "")).strip()
        article_text = normalize_text(article.get("text") or article.get("source_text") or "")
        if not article_number or not article_text:
            continue
        records.append(
            build_record(
                source,
                {
                    "article": article_number.lower(),
                    "pages": [],
                    "text": article_text,
                },
            )
        )
    return records


def min_article_count_for_ready(source: TreatySource) -> int:
    if source.variant.startswith("protokol"):
        return 3
    if source.variant == "tekst_syntetyczny_mli":
        return 10
    return 10


def build_outputs(sources: list[TreatySource]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for source in sources:
        status = "pending_ocr"
        pages: list[dict[str, Any]] = []
        extracted_chars = 0
        article_count = 0
        extraction_method = "none"
        structured_records = load_structured_json_records(source)
        if structured_records:
            records.extend(structured_records)
            manifest.append(
                {
                    "country": source.country,
                    "slug": source.slug,
                    "variant": source.variant,
                    "pdf_path": str(source.pdf_path),
                    "source_url": source.source_url,
                    "status": "ready",
                    "extraction_method": "structured_json",
                    "extracted_chars": sum(len(record["content_text"]) for record in structured_records),
                    "article_count": len(structured_records),
                    "included_in_jsonl": True,
                }
            )
            continue
        pdf_path = source.pdf_path if source.pdf_path.is_absolute() else REPO_ROOT / source.pdf_path
        if pdf_path.exists():
            pages = extract_pdf_text_pages(source)
            extracted_chars = sum(len(page["text"]) for page in pages)
            treaty_articles: list[dict[str, Any]] = []
            if extracted_chars >= 3000 and source.ready_without_ocr:
                extraction_method = "pdf_text"
                treaty_articles = list(iter_article_records(source, pages))
                article_count = len(treaty_articles)
                if article_count >= min_article_count_for_ready(source):
                    status = "ready"
                    for article in treaty_articles:
                        records.append(build_record(source, article))
                else:
                    status = "partial_text_only"
            if status != "ready":
                ocr_candidate = ocr_pages(source)
                ocr_chars = sum(len(page["text"]) for page in ocr_candidate)
                if ocr_chars >= 3000:
                    pages = ocr_candidate
                    extracted_chars = ocr_chars
                    extraction_method = "ocr"
                    treaty_articles = list(iter_article_records(source, pages))
                    article_count = len(treaty_articles)
                    if article_count >= min_article_count_for_ready(source):
                        status = "ready"
                        for article in treaty_articles:
                            records.append(build_record(source, article))
                    else:
                        status = "partial_text_only"
                elif extracted_chars >= 3000 and not source.ready_without_ocr:
                    status = "text_unavailable_in_pdf_layer"
                elif extracted_chars > 0 or ocr_chars > 0:
                    extraction_method = "ocr" if ocr_chars >= extracted_chars and ocr_chars > 0 else extraction_method
                    extracted_chars = max(extracted_chars, ocr_chars)
                    status = "partial_text_only"
        manifest.append(
            {
                "country": source.country,
                "slug": source.slug,
                "variant": source.variant,
                "pdf_path": str(source.pdf_path),
                "source_url": source.source_url,
                "status": status,
                "extraction_method": extraction_method,
                "extracted_chars": extracted_chars,
                "article_count": article_count,
                "included_in_jsonl": status == "ready",
            }
        )

    occurrences = Counter(record["document_id"] for record in records)
    seen: defaultdict[str, int] = defaultdict(int)
    for record in records:
        document_id = record["document_id"]
        if occurrences[document_id] > 1:
            seen[document_id] += 1
            record["document_id"] = f"{document_id}-occurrence-{seen[document_id]}"
    return records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    records, manifest = build_outputs(list(CORE_TREATY_SOURCES))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "manifest_entries": len(manifest)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
