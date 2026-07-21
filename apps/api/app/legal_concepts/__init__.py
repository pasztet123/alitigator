"""Versioned, generic legal-concept taxonomy and deterministic matcher."""

from .loader import ConceptDictionary, load_default_dictionary
from .matcher import ConceptMatch, ConceptMatchResult, ConceptMatcher
from .schema import ConceptDefinition

__all__ = [
    "ConceptDefinition", "ConceptDictionary", "ConceptMatch", "ConceptMatchResult",
    "ConceptMatcher", "load_default_dictionary",
]
