"""Build provision-level UFR JSONL from the structured official ELI HTML."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path


SOURCE_URL = "https://api.sejm.gov.pl/eli/acts/DU/2023/326/text.html"
PDF_URL = "https://eli.gov.pl/api/acts/DU/2023/326/text/U/D20230326Lj.pdf"
ACT_TITLE = "Ustawa z dnia 26 stycznia 2023 r. o fundacji rodzinnej"
PUBLICATION = "Dz.U. 2023 poz. 326, 825"
LEGAL_STATE_DATE = "2023-10-17"
PUBLISHED_DATE = "2023-02-21"
DOCUMENT_PREFIX = "pl-ustawa-o-fundacji-rodzinnej-2023-326-art.-"
ARTICLE_ID_RE = re.compile(r"(?:^|-)arti_(\d+)$")
WHITESPACE_RE = re.compile(r"\s+")
CHILD_MARKER_RE = re.compile(r"^(?:\d+[a-z]*\.|\d+[a-z]*\)|[a-z]\)|§\s*\d+[a-z]*\.)$", re.IGNORECASE)


def _normalize(value: str) -> str:
    return WHITESPACE_RE.sub(" ", value.replace("\u00ad", "").replace("\u00a0", " ")).strip()


def _join_child_markers(text: str) -> str:
    """Join ELI's separate ``h3`` marker with its following unit body."""

    lines = [line for line in text.splitlines() if line.strip()]
    result: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if (
            CHILD_MARKER_RE.fullmatch(line)
            and index + 1 < len(lines)
            and not CHILD_MARKER_RE.fullmatch(lines[index + 1])
            and not re.fullmatch(r"Art\.\s*\d+[a-z]*\.", lines[index + 1], re.IGNORECASE)
        ):
            result.append(f"{line} {lines[index + 1]}")
            index += 2
            continue
        result.append(line)
        index += 1
    return "\n".join(result)


class EliArticleParser(HTMLParser):
    """Extract only top-level article units and omit display-only footnotes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.article_number: str | None = None
        self.article_depth: int | None = None
        self.lines: list[str] = []
        self.capture_depth: int | None = None
        self.capture_parts: list[str] = []
        self.skip_depth: int | None = None
        self.articles: dict[int, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.stack.append(tag)
        attributes = {key.casefold(): value or "" for key, value in attrs}
        classes = set(attributes.get("class", "").casefold().split())
        element_id = attributes.get("id", "")

        if self.skip_depth is None and ({"xhidden", "tooltip-text"} & classes or tag == "sup"):
            self.skip_depth = len(self.stack)

        if self.article_number is None and tag == "div" and "unit_arti" in classes:
            match = ARTICLE_ID_RE.search(element_id)
            if match:
                self.article_number = match.group(1)
                self.article_depth = len(self.stack)
                self.lines = []

        if self.article_number is not None and self.skip_depth is None:
            if tag == "h3" or (tag == "div" and attributes.get("data-template", "").casefold() == "xtext"):
                self.capture_depth = len(self.stack)
                self.capture_parts = []

    def handle_data(self, data: str) -> None:
        if self.capture_depth is not None and self.skip_depth is None:
            self.capture_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        depth = len(self.stack)
        if self.capture_depth == depth:
            line = _normalize(" ".join(self.capture_parts))
            if line:
                self.lines.append(line)
            self.capture_depth = None
            self.capture_parts = []
        if self.skip_depth == depth:
            self.skip_depth = None
        if self.article_depth == depth and self.article_number is not None:
            number = int(self.article_number)
            if number in self.articles:
                raise ValueError(f"duplicate top-level ELI article: {number}")
            self.articles[number] = "\n".join(self.lines).strip()
            self.article_number = None
            self.article_depth = None
            self.lines = []
        if self.stack:
            self.stack.pop()


def build_records(html: str) -> list[dict]:
    sys.path.insert(0, str(Path("apps/api").resolve()))
    from app.law_chunk import CHUNKER_VERSION, build_provision_units

    parser = EliArticleParser()
    parser.feed(html)
    expected = set(range(1, 146))
    found = set(parser.articles)
    if found != expected:
        raise ValueError(
            f"ELI UFR article audit failed; missing={sorted(expected - found)}, extra={sorted(found - expected)}"
        )

    records: list[dict] = []
    for number in sorted(parser.articles):
        content = _join_child_markers(parser.articles[number])
        expected_heading = f"Art. {number}."
        if not content.startswith(expected_heading):
            raise ValueError(f"article {number} has an invalid heading: {content[:60]!r}")
        document_id = f"{DOCUMENT_PREFIX}{number}"
        units = [
            unit
            for unit in build_provision_units(
                content,
                article_document_id=document_id,
                record_document_id=document_id,
                article_hint=f"art. {number}",
            )
            if unit.get("article") == str(number)
        ]
        # Articles 129-139 amend other acts and contain quoted editorial
        # markers from those acts.  Their only UFR-level address is the
        # top-level article, so expose the full article as one exact unit
        # instead of fabricating UFR sub-units from the quoted legislation.
        if 129 <= number <= 139:
            top = next(unit for unit in units if unit["citation"] == f"art. {number}")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            units = [
                {
                    **top,
                    "source_span_start": 0,
                    "source_span_end": len(content),
                    "text": content,
                    "content_sha256": content_hash,
                    "content_hash": content_hash,
                }
            ]
        records.append(
            {
                "document_id": document_id,
                "article_document_id": document_id,
                "source": "eli",
                "source_type": "statute",
                "source_subtype": "consolidated_text",
                "authority": "Kancelaria Sejmu / Marszałek Sejmu RP",
                "jurisdiction": "PL",
                "act_title": ACT_TITLE,
                "publication": PUBLICATION,
                "legal_state_date": LEGAL_STATE_DATE,
                "published_date": PUBLISHED_DATE,
                "subject": f"O fundacji rodzinnej - art. {number}",
                "legal_provisions": [f"art. {number}"],
                "issues": ["ufr"],
                "law_tags": ["UFR", PUBLICATION],
                "source_url": SOURCE_URL,
                "source_pdf": PDF_URL,
                "source_pages": [],
                "hierarchy": {},
                "pre_chunked": True,
                "content_text": content,
                "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "chunker_version": CHUNKER_VERSION,
                "provision_units": units,
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="apps/api/data/laws/processed/family_foundation_primary_bundle.jsonl",
    )
    parser.add_argument("--html-path", default="")
    args = parser.parse_args()

    if args.html_path:
        html = Path(args.html_path).read_text(encoding="utf-8")
    else:
        with urllib.request.urlopen(SOURCE_URL, timeout=30) as response:
            html = response.read().decode("utf-8")
    records = build_records(html)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    print({"output": str(output), "records": len(records)}, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
