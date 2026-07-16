"""Extract and structurally chunk a Polish consolidated act published as a PDF.

The output is JSONL intentionally separate from the EUREKA corpus.  Each record
represents an article (or a coherent part of a long article), retaining its
division/chapter context and source pages.  It can be indexed as a distinct
legal-source collection once the RAG is wired to search statutes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pypdf import PdfReader


DIVISION_RE = re.compile(r"^(DZIAŁ|ROZDZIAŁ|ODDZIAŁ)\s+([IVXLCDM]+|\d+[A-Z]?)\b.*$", re.IGNORECASE)
ARTICLE_LINE_RE = re.compile(r"^Art\.\s*\d+[a-z]*\.(?=\s|\d|\(|$)")
# A lower-case "załącznik nr" is often an in-text reference to an annex; only
# the capitalized page heading starts the actual annex section.
ANNEX_START_RE = re.compile(r"^Załącznik(?:i)?(?:\s+nr|\s+do ustawy)")
PARAGRAPH_RE = re.compile(r"(?=^\s*\d+[a-z]*\.\s+)", re.MULTILINE)
HEADER_RE = re.compile(r"^(?:©Kancelaria Sejmu\s+s\.\s*\d+/\d+|Dziennik Ustaw\s+[–-]\s+\d+\s+[–-]\s+Poz\.\s+\d+|\d{4}-\d{2}-\d{2})\s*$", re.MULTILINE)
WHITESPACE_RE = re.compile(r"[ \t]+")
BLANKS_RE = re.compile(r"\n{3,}")

CHUNKER_VERSION = "provision_units_v1"
_ARTICLE_WORD_RE = re.compile(r"(?im)^(?P<indent>[ \t]*)artyku(?:ł|l|∏)\b")
_INLINE_ARTICLE_CHILD_RE = re.compile(
    r"(?im)^(?P<parent>[ \t]*art\.[ \t]*\d+[a-z]*\.?)"
    r"(?P<gap>[ \t]+)(?=(?:(?:ust\.[ \t]*)?\d+[a-z]*\.[ \t]+|§[ \t]*\d))"
)
_INLINE_PARAGRAPH_SECTION_RE = re.compile(
    r"(?im)^(?P<parent>[ \t]*§[ \t]*\d+[a-z]*\.?)"
    r"(?P<gap>[ \t]+)(?=(?:ust\.[ \t]*)?\d+[a-z]*\.[ \t]+)"
)
_INLINE_SECTION_POINT_RE = re.compile(
    r"(?im)^(?P<parent>[ \t]*(?:ust\.[ \t]*)?\d+[a-z]*\.)"
    r"(?P<gap>[ \t]+)(?=(?:pkt[ \t]*)?\d+[a-z]*\)[ \t]+)"
)
_INLINE_POINT_LETTER_RE = re.compile(
    r"(?im)^(?P<parent>[ \t]*(?:pkt[ \t]*)?\d+[a-z]*\))"
    r"(?P<gap>[ \t]+)(?=(?:lit\.[ \t]*)?[a-z]\)[ \t]+)"
)
_PROVISION_ARTICLE_RE = re.compile(r"^\s*art\.\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_PROVISION_PARAGRAPH_RE = re.compile(r"^\s*§\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_PROVISION_EXPLICIT_SECTION_RE = re.compile(r"^\s*ust\.\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_PROVISION_EXPLICIT_POINT_RE = re.compile(r"^\s*pkt\s*(\d+[a-z]*)\.?\s*(.*)$", re.IGNORECASE)
_PROVISION_EXPLICIT_LETTER_RE = re.compile(r"^\s*lit\.\s*([a-z])\)?\s*(.*)$", re.IGNORECASE)
_PROVISION_SECTION_RE = re.compile(r"^\s*(\d+[a-z]*)\.\s+(.+)$", re.IGNORECASE)
_PROVISION_POINT_RE = re.compile(r"^\s*(\d+[a-z]*)\)\s*(.+)$", re.IGNORECASE)
_PROVISION_LETTER_RE = re.compile(r"^\s*([a-z])\)\s*(.+)$", re.IGNORECASE)
_SOURCE_NOTE_RE = re.compile(
    r"^\s*\d+\)\s*(?:"
    r"Niniejsz[ąa]\s+ustaw[ąa]|"
    r"Zmiany\s+tekstu\s+jednolitego|"
    r"Zmiana\s+wymienionej\s+ustawy|"
    r"W\s+brzmieniu\s+ustalonym|"
    r"Dodany\s+przez|"
    r"Uchylony\s+przez"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PageText:
    number: int
    text: str


def _replace_gap_with_newline(match: re.Match[str]) -> str:
    """Expose an inline child marker without changing any source offsets."""
    gap = match.group("gap")
    return f"{match.group('parent')}\n{' ' * (len(gap) - 1)}"


def _provision_parser_view(text: str) -> str:
    """Return a same-length parser view for common PDF/OCR legal markers.

    ``ProvisionParser`` understands ``Art.`` while treaty PDFs normally use
    ``Artykuł``.  PDF extraction can also put an article and its first child on
    one line.  Both normalizations below preserve the number of characters, so
    offsets produced against this view remain offsets into the source text.
    """

    def replace_article_word(match: re.Match[str]) -> str:
        marker = match.group(0)
        indent = match.group("indent")
        return f"{indent}Art.{' ' * (len(marker) - len(indent) - 4)}"

    parser_text = _ARTICLE_WORD_RE.sub(replace_article_word, text)
    for pattern in (
        _INLINE_ARTICLE_CHILD_RE,
        _INLINE_PARAGRAPH_SECTION_RE,
        _INLINE_SECTION_POINT_RE,
        _INLINE_POINT_LETTER_RE,
    ):
        parser_text = pattern.sub(_replace_gap_with_newline, parser_text)
    assert len(parser_text) == len(text)
    return parser_text


def _format_provision_citation(context: dict[str, str | None]) -> str:
    parts: list[str] = []
    for key, label in (
        ("article", "art."),
        ("paragraph", "§"),
        ("section", "ust."),
        ("point", "pkt"),
        ("letter", "lit."),
    ):
        if context[key]:
            parts.append(f"{label} {context[key]}")
    return " ".join(parts)


def _provision_id(document_id: str, context: dict[str, str | None]) -> str:
    path = _format_provision_citation(context) or "document"
    digest = hashlib.sha256(f"{document_id}\0current\0{path}".encode("utf-8")).hexdigest()[:16]
    return f"{document_id}:current:{digest}"


def _parse_provision_marker(
    line: str,
    context: dict[str, str | None],
) -> tuple[str, str] | None:
    for level, pattern in (
        ("article", _PROVISION_ARTICLE_RE),
        ("paragraph", _PROVISION_PARAGRAPH_RE),
        ("section", _PROVISION_EXPLICIT_SECTION_RE),
        ("point", _PROVISION_EXPLICIT_POINT_RE),
        ("letter", _PROVISION_EXPLICIT_LETTER_RE),
    ):
        match = pattern.match(line)
        if match:
            return level, match.group(1).casefold()
    match = _PROVISION_SECTION_RE.match(line)
    if match and (context["article"] or context["paragraph"]):
        return "section", match.group(1).casefold()
    match = _PROVISION_POINT_RE.match(line)
    if match and (context["section"] or context["paragraph"]):
        return "point", match.group(1).casefold()
    match = _PROVISION_LETTER_RE.match(line)
    if match and context["point"]:
        return "letter", match.group(1).casefold()
    return None


def _advance_provision_context(
    context: dict[str, str | None],
    level: str,
    value: str,
) -> None:
    if level == "article":
        context.update(article=value, paragraph=None, section=None, point=None, letter=None)
    elif level == "paragraph":
        context.update(paragraph=value, section=None, point=None, letter=None)
    elif level == "section":
        context.update(section=value, point=None, letter=None)
    elif level == "point":
        context.update(point=value, letter=None)
    else:
        context["letter"] = value


def _parent_provision_id(
    document_id: str,
    context: dict[str, str | None],
    level: str,
) -> str | None:
    if level == "article":
        return None
    parent = dict(context)
    if level == "paragraph":
        parent["paragraph"] = None
    elif level == "section":
        parent["section"] = None
    elif level == "point":
        parent["point"] = None
    else:
        parent["letter"] = None
    return _provision_id(document_id, parent)


def _parse_provision_spans(text: str, document_id: str) -> list[dict]:
    """Small CLI-safe adapter mirroring the v2 provision marker semantics."""
    context: dict[str, str | None] = {
        "article": None,
        "paragraph": None,
        "section": None,
        "point": None,
        "letter": None,
    }
    result: list[dict] = []
    current: dict | None = None

    def finish(end: int) -> None:
        nonlocal current
        if current is not None:
            current["source_span_end"] = max(current["source_span_start"], end)
            result.append(current)
            current = None

    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        marker = _parse_provision_marker(line, context)
        if marker is not None:
            finish(offset)
            level, value = marker
            _advance_provision_context(context, level, value)
            current = {
                "provision_id": _provision_id(document_id, context),
                "version_id": "current",
                "unit_type": {
                    "article": "article",
                    "paragraph": "paragraph",
                    "section": "section",
                    "point": "point",
                    "letter": "letter",
                }[level],
                "citation": _format_provision_citation(context),
                "article": context["article"],
                "paragraph": context["paragraph"],
                "section": context["section"],
                "point": context["point"],
                "letter": context["letter"],
                "parent_id": _parent_provision_id(document_id, context, level),
                "source_span_start": offset,
            }
        offset += len(raw_line)
    finish(len(text))
    return result


def build_provision_units(
    text: str,
    *,
    article_document_id: str,
    record_document_id: str,
    article_hint: str | None = None,
    source_offset: int = 0,
) -> list[dict]:
    """Build stable, source-addressable metadata for one article record.

    ``article_hint`` seeds the parser when ``text`` is a later part of a long
    article and therefore starts at an ``ust.`` marker.  The seed is not
    emitted and is subtracted from all spans.  Unit spans are half-open and
    always address the exact unit text in the record's ``content_text``.
    """
    if not text.strip():
        return []

    parse_source = text
    seed_length = 0
    if article_hint and not re.match(r"^\s*(?:art\.|artyku(?:ł|l|∏)\b)", text, re.IGNORECASE):
        article_number = article_hint.removeprefix("art. ").strip()
        seed = f"Art. {article_number}.\n"
        parse_source = f"{seed}{text}"
        seed_length = len(seed)

    parsed = _parse_provision_spans(_provision_parser_view(parse_source), article_document_id)
    parsed_ids = {unit["provision_id"] for unit in parsed}
    result: list[dict] = []
    for unit in parsed:
        if unit["source_span_start"] < seed_length:
            continue
        start = unit["source_span_start"]
        end = unit["source_span_end"]
        while start < end and parse_source[start].isspace():
            start += 1
        while end > start and parse_source[end - 1].isspace():
            end -= 1
        if start >= end:
            continue
        local_start = start - seed_length
        local_end = end - seed_length
        unit_text = text[local_start:local_end]
        # Kancelaria Sejmu PDFs place numbered editorial footnotes in the
        # article text flow.  Their markers look exactly like statutory
        # points (for example ``2) Zmiany tekstu jednolitego...``).  They are
        # source metadata, not editorial units, and indexing them as a point
        # creates duplicate/fictitious citations.
        if _SOURCE_NOTE_RE.match(unit_text):
            continue
        content_hash = hashlib.sha256(unit_text.encode("utf-8")).hexdigest()
        result.append(
            {
                "provision_id": unit["provision_id"],
                "document_id": article_document_id,
                "record_document_id": record_document_id,
                "version_id": unit["version_id"],
                "unit_type": unit["unit_type"],
                "citation": unit["citation"],
                "article": unit["article"],
                "paragraph": unit["paragraph"],
                "section": unit["section"],
                "point": unit["point"],
                "letter": unit["letter"],
                "parent_id": unit["parent_id"] if unit["parent_id"] in parsed_ids else None,
                "source_span_start": source_offset + local_start,
                "source_span_end": source_offset + local_end,
                "text": unit_text,
                "content_sha256": content_hash,
                "content_hash": content_hash,
            }
        )
    return result


def normalize(text: str) -> str:
    text = text.replace("\u00ad", "").replace("\u00a0", " ")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = HEADER_RE.sub("", text)
    text = "\n".join(WHITESPACE_RE.sub(" ", line).strip() for line in text.splitlines())
    return BLANKS_RE.sub("\n\n", text).strip()


def extract_act_pages(pdf_path: Path, *, short_title: str) -> list[PageText]:
    reader = PdfReader(pdf_path)
    pages = [PageText(index + 1, normalize(page.extract_text() or "")) for index, page in enumerate(reader.pages)]
    act_start_re = re.compile(
        r"\bU\s*S\s*T\s*A\s*W\s*A\s+z dnia .*?\b" + re.escape(short_title),
        re.DOTALL | re.IGNORECASE,
    )
    for index, page in enumerate(pages):
        match = act_start_re.search(page.text)
        if match:
            return [PageText(page.number, page.text[match.start() :]), *pages[index + 1 :]]
    raise ValueError("Could not locate the beginning of the act in the supplied PDF")


def article_title(text: str) -> str:
    match = re.match(r"Art\.\s*(\d+[a-z]*)\.(?=\s|\d|\(|$)", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Article without an article number: {text[:80]!r}")
    return f"art. {match.group(1)}"


def split_long_article(text: str, target_chars: int) -> list[str]:
    """Split only before numbered paragraphs; never cut a legal sentence arbitrarily."""
    if len(text) <= target_chars:
        return [text]
    pieces = [piece.strip() for piece in PARAGRAPH_RE.split(text) if piece.strip()]
    if len(pieces) == 1:
        return [text]
    result: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}\n{piece}".strip() if current else piece
        if current and len(candidate) > target_chars:
            result.append(current)
            current = piece
        else:
            current = candidate
    if current:
        result.append(current)
    return result


def chunks_from_pages(pages: list[PageText], target_chars: int) -> Iterator[dict]:
    """Walk the extracted lines in source order, retaining legal hierarchy."""
    context: dict[str, str] = {}
    article_lines: list[str] = []
    article_pages: list[int] = []
    in_annex = False

    def flush_article() -> Iterator[dict]:
        nonlocal article_lines, article_pages
        article = "\n".join(article_lines).strip()
        if article:
            yield from article_records(article, context, article_pages, target_chars)
        article_lines, article_pages = [], []

    for page in pages:
        for line in page.text.splitlines():
            line = line.strip()
            if not line:
                continue
            if ANNEX_START_RE.match(line):
                yield from flush_article()
                in_annex = True
                continue
            if in_annex:
                continue
            heading = DIVISION_RE.match(line)
            if heading:
                yield from flush_article()
                kind = heading.group(1).lower()
                context[kind] = line
                if kind == "dział":
                    context.pop("rozdział", None)
                    context.pop("oddział", None)
                elif kind == "rozdział":
                    context.pop("oddział", None)
                continue
            if ARTICLE_LINE_RE.match(line):
                yield from flush_article()
                article_lines = [line]
                article_pages = [page.number]
                continue
            if article_lines:
                article_lines.append(line)
                article_pages.append(page.number)

    yield from flush_article()


def article_records(article: str, context: dict[str, str], pages: list[int], target_chars: int) -> Iterator[dict]:
    provision = article_title(article)
    parts = split_long_article(article, target_chars)
    for part_index, text in enumerate(parts, start=1):
        yield {
            "provision": provision,
            "part_index": part_index,
            "part_count": len(parts),
            "text": text,
            "pages": sorted(set(pages)),
            "hierarchy": dict(context),
        }


def build_records(
    pdf_path: Path,
    *,
    target_chars: int,
    source_url: str,
    law_id: str,
    short_title: str,
    act_title: str,
    publication: str,
    legal_state_date: str,
    published_date: str,
    tax_tag: str,
    source_subtype: str = "consolidated_text",
) -> list[dict]:
    records = []
    for chunk in chunks_from_pages(extract_act_pages(pdf_path, short_title=short_title), target_chars):
        content = "\n\n".join([*chunk["hierarchy"].values(), chunk["text"]])
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        suffix = f"-part-{chunk['part_index']}" if chunk["part_count"] > 1 else ""
        article_document_id = f"pl-{law_id}-{chunk['provision'].replace(' ', '-')}"
        record_document_id = f"{article_document_id}{suffix}"
        text_offset = len(content) - len(chunk["text"])
        records.append(
            {
                "document_id": record_document_id,
                "article_document_id": article_document_id,
                "source": "eli",
                "source_type": "statute",
                "source_subtype": source_subtype,
                "authority": "Kancelaria Sejmu / Marszałek Sejmu RP",
                "jurisdiction": "PL",
                "act_title": act_title,
                "publication": publication,
                "legal_state_date": legal_state_date,
                "published_date": published_date,
                "subject": f"{short_title.capitalize()} - {chunk['provision']}",
                "legal_provisions": [chunk["provision"]],
                "issues": [tax_tag.lower()],
                "law_tags": [tax_tag.upper(), publication],
                "source_url": source_url,
                "source_pdf": str(pdf_path),
                "source_pages": chunk["pages"],
                "hierarchy": chunk["hierarchy"],
                "pre_chunked": True,
                "content_text": content,
                "content_sha256": digest,
                "chunker_version": CHUNKER_VERSION,
                "provision_units": build_provision_units(
                    chunk["text"],
                    article_document_id=article_document_id,
                    record_document_id=record_document_id,
                    article_hint=chunk["provision"],
                    source_offset=text_offset,
                ),
            }
        )
    occurrences = Counter(record["document_id"] for record in records)
    seen: defaultdict[str, int] = defaultdict(int)
    for record in records:
        document_id = record["document_id"]
        if occurrences[document_id] > 1:
            seen[document_id] += 1
            record["document_id"] = f"{document_id}-occurrence-{seen[document_id]}"
        for unit in record["provision_units"]:
            unit["record_document_id"] = record["document_id"]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-chars", type=int, default=2800)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--law-id", default="ustawa-o-podatku-akcyzowym-2026-412")
    parser.add_argument("--short-title", default="o podatku akcyzowym")
    parser.add_argument("--act-title", default="Ustawa z dnia 6 grudnia 2008 r. o podatku akcyzowym")
    parser.add_argument("--publication", default="Dz.U. 2026 poz. 412")
    parser.add_argument("--legal-state-date", default="2026-03-10")
    parser.add_argument("--published-date", default="2026-03-30")
    parser.add_argument("--tax-tag", default="AKCYZA")
    parser.add_argument(
        "--source-subtype",
        default="consolidated_text",
        choices=("consolidated_text", "codified_text"),
    )
    args = parser.parse_args()
    records = build_records(
        args.pdf,
        target_chars=args.target_chars,
        source_url=args.source_url,
        law_id=args.law_id,
        short_title=args.short_title,
        act_title=args.act_title,
        publication=args.publication,
        legal_state_date=args.legal_state_date,
        published_date=args.published_date,
        tax_tag=args.tax_tag,
        source_subtype=args.source_subtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
