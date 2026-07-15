"""Capture deterministic article text for UPO PDFs whose layout is not parsable.

This is a *maintenance-time* importer.  The application never requests the
helper pages at runtime.  Generated files retain the official MF PDF URL as
the legal source used by retrieval; the helper URL is audit-only provenance
for the transcription that restores the article boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTICLE_RE = re.compile(r"(?im)^\s*Artykuł\s+(\d+)\s*$")
WIDGET_RE = re.compile(r"\n(?:Istniejące wersje czasowe|PRZEPISY\.|GOFIN WYJAŚNIA:).*", re.DOTALL)
PAGE_CHROME_RE = re.compile(r"\n(?:A\n){3}Pełna treść\naktu prawnego.*", re.DOTALL)
SWISS_OLD_ARTICLE_15 = (
    "Bez względu na poprzednie postanowienia niniejszego artykułu wynagrodzenie "
    "uzyskiwane przez osobę z tytułu pracy najemnej wykonywanej na pokładzie statku, "
    "samolotu eksploatowanego w komunikacji międzynarodowej lub na pokładzie barki "
    "wykorzystywanej w transporcie na wodach śródlądowych może podlegać opodatkowaniu "
    "tylko w tym Umawiającym się Państwie, w którym znajduje się miejsce rzeczywistego "
    "zarządu przedsiębiorstwa."
)
SWISS_CORRECTED_ARTICLE_15 = (
    "Bez względu na poprzednie postanowienia niniejszego artykułu, wynagrodzenie "
    "uzyskiwane przez osobę z tytułu pracy najemnej wykonywanej na pokładzie statku, "
    "samolotu eksploatowanego w komunikacji międzynarodowej lub na pokładzie barki "
    "wykorzystywanej w transporcie na wodach śródlądowych może podlegać opodatkowaniu "
    "w tym Umawiającym się Państwie, w którym znajduje się miejsce rzeczywistego "
    "zarządu przedsiębiorstwa."
)


@dataclass(frozen=True)
class Target:
    slug: str
    variant: str
    helper_url: str
    official_pdf_url: str
    output: Path
    expected_articles: int
    correction_url: str | None = None


TARGETS = (
    Target(
        slug="hiszpania",
        variant="umowa",
        helper_url="https://przepisy.gofin.pl/przepisy,4,15,123,95,,,umowa-miedzy-rzadem-polskiej-rzeczypospolitej-ludowej.html",
        official_pdf_url="https://www.podatki.gov.pl/media/0yppho2i/hiszpania-konwencja-tekst-polski.pdf",
        output=Path("resources/upo/hiszpania/upo_polska_hiszpania_pl.json"),
        expected_articles=30,
    ),
    Target(
        slug="szwajcaria",
        variant="umowa",
        helper_url="https://przepisy.gofin.pl/przepisy%2C4%2C15%2C82%2C178%2C%2C19930323%2Ckonwencja-miedzy-rzeczapospolita-polska-a-konfederacja.html",
        official_pdf_url="https://www.podatki.gov.pl/media/fbol1ik4/19910902_konwencja_szwajcaria_pl_e.pdf",
        output=Path("resources/upo/szwajcaria/upo_polska_szwajcaria_pl.json"),
        expected_articles=28,
        correction_url="https://eli.gov.pl/api/acts/DU/2021/262/text.html",
    ),
    Target(
        slug="usa",
        variant="umowa_1974",
        helper_url="https://przepisy.gofin.pl/przepisy,4,15,84,174,,19760918,umowa-miedzy-rzadem-polskiej-rzeczypospolitej-ludowej-a.html",
        official_pdf_url="https://www.podatki.gov.pl/media/fygbbhi0/19741008_usa_konwencja_tekst_polski.pdf",
        output=Path("resources/upo/usa/upo_polska_usa_1974_pl.json"),
        expected_articles=26,
    ),
    Target(
        slug="wielka_brytania",
        variant="umowa",
        helper_url="https://przepisy.gofin.pl/przepisy,4,15,73,196,,,konwencja-miedzy-rzeczapospolita-polska-a-zjednoczonym.html",
        official_pdf_url="https://www.podatki.gov.pl/media/uanpfvts/wlk-brytania-konwencja-tekst-polski.pdf",
        output=Path("resources/upo/wielka_brytania/upo_polska_wielka_brytania_pl.json"),
        expected_articles=29,
    ),
)


class TextCollector(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "center", "h1", "h2", "h3", "h4", "li", "tr"}

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS - {"br"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def download(url: str) -> str:
    request = Request(url, headers={"User-Agent": "aLitigator treaty-maintenance/1.0"})
    with urlopen(request, timeout=30) as response:  # nosec B310 - fixed HTTPS sources above
        return response.read().decode("utf-8", errors="replace")


def html_to_lines(html: str) -> str:
    parser = TextCollector()
    parser.feed(html)
    text = "".join(parser.parts).replace("\xa0", " ")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def article_units(text: str, target: Target) -> list[dict[str, object]]:
    matches = list(ARTICLE_RE.finditer(text))
    articles: list[dict[str, object]] = []
    seen: set[int] = set()
    for index, match in enumerate(matches):
        article_number = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = PAGE_CHROME_RE.sub("", WIDGET_RE.sub("", text[match.end() : end])).strip()
        # The published Swiss transcription carries the historical typo in its
        # heading.  The official correction (Dz.U. 2021 poz. 262) makes this
        # the second, distinct art. 15 rather than a second art. 14.
        if target.slug == "szwajcaria" and article_number == 14 and "Praca najemna" in body:
            article_number = 15
            body = body.replace(SWISS_OLD_ARTICLE_15, SWISS_CORRECTED_ARTICLE_15)
        if article_number in seen:
            raise ValueError(f"duplicate article {article_number} in {target.slug}/{target.variant}")
        seen.add(article_number)
        if not body:
            raise ValueError(f"empty article {article_number} in {target.slug}/{target.variant}")
        articles.append({"article_number": article_number, "text": body, "source_text": body})
    expected = list(range(1, target.expected_articles + 1))
    found = [int(article["article_number"]) for article in articles]
    if found != expected:
        raise ValueError(f"{target.slug}/{target.variant}: expected {expected}, got {found}")
    return articles


def write_target(target: Target) -> dict[str, object]:
    raw_html = download(target.helper_url)
    articles = article_units(html_to_lines(raw_html), target)
    payload = {
        "schema_version": 1,
        "document_type": "tax_treaty_article_transcription",
        "language": "pl",
        "source": {
            "official_pdf_url": target.official_pdf_url,
            "transcription_helper_url": target.helper_url,
            "official_correction_url": target.correction_url,
            "transcription_sha256": hashlib.sha256(
                "\n".join(str(article["text"]) for article in articles).encode("utf-8")
            ).hexdigest(),
        },
        "articles": articles,
    }
    output = REPO_ROOT / target.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(target.output), "article_count": len(articles)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slug", action="append", choices=[target.slug for target in TARGETS])
    args = parser.parse_args()
    selected = [target for target in TARGETS if not args.slug or target.slug in args.slug]
    results = [write_target(target) for target in selected]
    print(json.dumps({"generated": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
