"""Deterministic recognition of named Polish tax-law institutions.

The package is deliberately independent from the model planner.  It turns a
versioned dictionary into traceable, bounded research inputs; it never answers
a legal question or decides whether a document is legally correct.
"""

from .dictionary import InstitutionDictionary, load_default_dictionary
from .matcher import InstitutionMatcher, InstitutionMatch, InstitutionMatchResult
from .merger import merge_locked_institutions

__all__ = [
    "InstitutionDictionary",
    "InstitutionMatch",
    "InstitutionMatcher",
    "InstitutionMatchResult",
    "load_default_dictionary",
    "merge_locked_institutions",
]
