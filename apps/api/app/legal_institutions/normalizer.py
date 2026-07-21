"""Conservative Polish text normalisation for deterministic dictionary matching."""

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


def normalize_polish(text: str) -> NormalizedText:
    """Normalise only formatting, never truncate words or strip morphology.

    The ASCII-folded form is a parallel comparison form for a user who omits
    diacritics.  It is not used for prefix matching, so ``vat`` cannot match a
    fragment of an unrelated word.
    """

    original = text or ""
    value = unicodedata.normalize("NFC", original).casefold()
    value = _HYPHEN_RE.sub("-", value)
    value = _SPACE_RE.sub(" ", value).strip()
    ascii_folded = "".join(
        char for char in unicodedata.normalize("NFD", value) if not unicodedata.combining(char)
    )
    tokens = tuple(_WORD_RE.findall(value))
    return NormalizedText(
        original=original,
        normalized=value,
        ascii_folded=ascii_folded,
        tokens=tokens,
    )


@lru_cache(maxsize=8_192)
def _phrase_patterns(phrase: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    normalized_phrase = normalize_polish(phrase)
    pattern = r"(?<!\w)" + re.escape(normalized_phrase.normalized).replace(r"\ ", r"\s+") + r"(?!\w)"
    ascii_pattern = r"(?<!\w)" + re.escape(normalized_phrase.ascii_folded).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.compile(pattern, flags=re.IGNORECASE), re.compile(ascii_pattern, flags=re.IGNORECASE)


def phrase_present(phrase: str, text: NormalizedText) -> bool:
    """Bounded phrase comparison with a diacritic-tolerant parallel form."""

    if not phrase or not phrase.strip():
        return False
    pattern, ascii_pattern = _phrase_patterns(phrase)
    if pattern.search(text.normalized):
        return True
    return bool(ascii_pattern.search(text.ascii_folded))


def phrase_tokens(phrase: str) -> tuple[str, ...]:
    return normalize_polish(phrase).tokens
