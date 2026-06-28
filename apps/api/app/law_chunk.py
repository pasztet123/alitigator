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


@dataclass(frozen=True)
class PageText:
    number: int
    text: str


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
        r"\bUSTAWA\s+z dnia .*?\b" + re.escape(short_title), re.DOTALL | re.IGNORECASE
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
) -> list[dict]:
    records = []
    for chunk in chunks_from_pages(extract_act_pages(pdf_path, short_title=short_title), target_chars):
        content = "\n\n".join([*chunk["hierarchy"].values(), chunk["text"]])
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        suffix = f"-part-{chunk['part_index']}" if chunk["part_count"] > 1 else ""
        records.append(
            {
                "document_id": f"pl-{law_id}-{chunk['provision'].replace(' ', '-')}{suffix}",
                "source": "eli",
                "source_type": "statute",
                "source_subtype": "consolidated_text",
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
            }
        )
    occurrences = Counter(record["document_id"] for record in records)
    seen: defaultdict[str, int] = defaultdict(int)
    for record in records:
        document_id = record["document_id"]
        if occurrences[document_id] > 1:
            seen[document_id] += 1
            record["document_id"] = f"{document_id}-occurrence-{seen[document_id]}"
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
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "records": len(records)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
