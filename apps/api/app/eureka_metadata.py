"""Deterministic metadata normalisation for Eureka interpretations.

Eureka exposes valuable classification data, but its statutory references use
the presentation format from the public search API (for example
``[VAT] ...-art. 86a-ust. 2``).  Retrieval receives user-facing citations in a
different form (``art. 86a``).  This module retains the source value and adds
only lossless, canonical aliases so both forms remain searchable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Sequence


_ARTICLE_RE = re.compile(
    r"\bart\.?\s*(?P<article>\d+[a-z]{0,4})\.?"
    r"(?:\s*(?:[-–—]\s*|\s+)ust\.?\s*(?P<paragraph>\d+[a-z]{0,4}))?"
    r"(?:\s*(?:[-–—]\s*|\s+)pkt\.?\s*(?P<point>\d+[a-z]{0,4}))?"
    r"(?:\s*(?:[-–—]\s*|\s+)lit\.?\s*(?P<letter>[a-z]))?",
    re.IGNORECASE,
)

# The tags are published by Eureka.  Map only tags with an unambiguous
# established tax domain; unknown special levies intentionally remain blank.
_TAG_DOMAINS = {
    "VAT": "VAT",
    "CIT": "CIT",
    "PIT": "PIT",
    "ZPDOF": "PIT",
    "PCC": "PCC",
    "PSD": "SD",
    "SD": "SD",
    "AKC": "AKCYZA",
    "AKCYZA": "AKCYZA",
    "OP": "ORDYNACJA",
    "ORDYNACJA": "ORDYNACJA",
    "UFR": "UFR",
}

_TEXT_DOMAIN_MARKERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("VAT", ("podatek od towarów i usług", "podatek od towarow i uslug", "ustawa o vat")),
    ("CIT", ("podatek dochodowy od osób prawnych", "podatek dochodowy od osob prawnych", "ustawa o cit")),
    ("PIT", (
        "podatek dochodowy od osób fizycznych",
        "podatek dochodowy od osob fizycznych",
        "zryczałtowany podatek dochodowy",
        "zryczaltowany podatek dochodowy",
        "ustawa o pit",
    )),
    ("PCC", ("podatek od czynności cywilnoprawnych", "podatek od czynnosci cywilnoprawnych", "ustawa o pcc")),
    ("SD", ("podatek od spadków i darowizn", "podatek od spadkow i darowizn")),
    ("AKCYZA", ("podatek akcyzowy", "ustawa o podatku akcyzowym")),
    ("ORDYNACJA", ("ordynacja podatkowa",)),
)


def _clean_values(values: Iterable[object]) -> list[str]:
    return [str(value).strip() for value in values if str(value).strip()]


def _deduplicate(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = value.strip()
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def derive_tax_domain(
    *,
    law_tags: Sequence[object] = (),
    legal_provisions: Sequence[object] = (),
    issues: Sequence[object] = (),
    subject: str = "",
    question_text: str = "",
    facts_text: str = "",
) -> str:
    """Return a conservative tax-domain label from source-owned metadata."""

    tagged_text = " ".join((*_clean_values(law_tags), *_clean_values(legal_provisions)))
    for tag in re.findall(r"\[([^\]]+)\]", tagged_text):
        domain = _TAG_DOMAINS.get(tag.strip().upper())
        if domain:
            return domain

    # ``issues`` and short structured sections are already in the local
    # corpus.  They are a safe fallback when Eureka omitted a recognised tag.
    text = " ".join(
        (*_clean_values(issues), subject, question_text[:1500], facts_text[:1500])
    ).casefold()
    for domain, markers in _TEXT_DOMAIN_MARKERS:
        if any(marker in text for marker in markers):
            return domain
    return ""


def canonical_provision_aliases(
    values: Sequence[object],
    *,
    tax_domain: str = "",
) -> list[str]:
    """Add canonical citations while preserving every source-provided value.

    Each alias comes from a citation already supplied by Eureka.  No provision
    is inferred from an answer, a benchmark signature, or a topic-specific
    rule.
    """

    original = _clean_values(values)
    aliases: list[str] = []
    fallback_domain = tax_domain.strip().upper()
    for value in original:
        value_domain = derive_tax_domain(legal_provisions=(value,)) or fallback_domain
        prefix = f"[{value_domain}] " if value_domain else ""
        for match in _ARTICLE_RE.finditer(value):
            pieces = [f"{prefix}art. {match.group('article').lower()}"]
            if match.group("paragraph"):
                pieces.append(f"ust. {match.group('paragraph').lower()}")
            if match.group("point"):
                pieces.append(f"pkt {match.group('point').lower()}")
            if match.group("letter"):
                pieces.append(f"lit. {match.group('letter').lower()}")
            aliases.append(" ".join(pieces))
    return _deduplicate((*original, *aliases))


@dataclass(frozen=True)
class InterpretationMetadata:
    tax_domain: str
    legal_provisions: tuple[str, ...]


def enrich_interpretation_metadata(
    *,
    tax_domain: str = "",
    law_tags: Sequence[object] = (),
    legal_provisions: Sequence[object] = (),
    issues: Sequence[object] = (),
    subject: str = "",
    question_text: str = "",
    facts_text: str = "",
) -> InterpretationMetadata:
    """Normalise only metadata that can be verified from the local record."""

    resolved_domain = tax_domain.strip().upper() or derive_tax_domain(
        law_tags=law_tags,
        legal_provisions=legal_provisions,
        issues=issues,
        subject=subject,
        question_text=question_text,
        facts_text=facts_text,
    )
    return InterpretationMetadata(
        tax_domain=resolved_domain,
        legal_provisions=tuple(
            canonical_provision_aliases(legal_provisions, tax_domain=resolved_domain)
        ),
    )
