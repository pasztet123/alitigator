from app.legal_rag_v2.authority import (
    HeuristicAuthorityExtractor,
    ModelAuthorityExtractor,
    SourceSpanValidationError,
    validate_authority_source_spans,
)


def validate_holding_span(source_text: str, start: int, end: int) -> str:
    """Fail closed for truncated or mid-token authority holdings."""
    if start < 0 or end <= start or end > len(source_text):
        raise SourceSpanValidationError("holding span is outside the document")
    if start and source_text[start - 1].isalnum() and source_text[start].isalnum():
        raise SourceSpanValidationError("holding starts in the middle of a word")
    holding = source_text[start:end].strip()
    if not holding or holding[-1] not in ".!?":
        raise SourceSpanValidationError("holding must end with a complete sentence")
    return holding

__all__ = [
    "HeuristicAuthorityExtractor", "ModelAuthorityExtractor",
    "SourceSpanValidationError", "validate_authority_source_spans", "validate_holding_span",
]
