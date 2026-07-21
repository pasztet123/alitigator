"""Single conservative normalisation path for question and document concepts."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

_WORD_RE = re.compile(r"[0-9A-Za-zĄĆĘŁŃÓŚŹŻąćęłńóśźż]+", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")
_HYPHEN_RE = re.compile(r"[‐‑‒–—―]")


@dataclass(frozen=True)
class NormalizedText:
    original: str
    normalized: str
    ascii_folded: str
    tokens: tuple[str, ...]


def normalize_text(value: str) -> NormalizedText:
    original = value or ""
    normalized = _SPACE_RE.sub(" ", _HYPHEN_RE.sub("-", unicodedata.normalize("NFC", original).casefold())).strip()
    ascii_folded = "".join(char for char in unicodedata.normalize("NFD", normalized) if not unicodedata.combining(char))
    return NormalizedText(original, normalized, ascii_folded, tuple(_WORD_RE.findall(normalized)))


@lru_cache(maxsize=16_384)
def _patterns(phrase: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    normal = normalize_text(phrase)
    expression = r"(?<!\w)" + re.escape(normal.normalized).replace(r"\ ", r"\s+") + r"(?!\w)"
    ascii_expression = r"(?<!\w)" + re.escape(normal.ascii_folded).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.compile(expression, re.I), re.compile(ascii_expression, re.I)


@lru_cache(maxsize=16_384)
def _phrase_forms(phrase: str) -> tuple[str, str]:
    normalized = normalize_text(phrase)
    return normalized.normalized, normalized.ascii_folded


def phrase_present(phrase: str, text: NormalizedText) -> bool:
    if not phrase or not phrase.strip():
        return False
    normalized_phrase, ascii_phrase = _phrase_forms(phrase)
    # Both inputs have their whitespace normalised.  This cheap containment
    # check preserves the regex's matching semantics while avoiding thousands
    # of full-document regex scans for concepts that cannot possibly occur.
    if normalized_phrase not in text.normalized and ascii_phrase not in text.ascii_folded:
        return False
    normal, ascii_pattern = _patterns(phrase)
    return bool(normal.search(text.normalized) or ascii_pattern.search(text.ascii_folded))
