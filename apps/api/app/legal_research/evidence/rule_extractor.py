from app.legal_research.models import LegalRuleEvidence


def validate_rule_span(rule: LegalRuleEvidence, provision_text: str) -> None:
    if rule.source_span_end > len(provision_text):
        raise ValueError("legal rule span is outside the controlling provision")
    if not provision_text[rule.source_span_start:rule.source_span_end].strip():
        raise ValueError("legal rule span is empty")


__all__ = ["LegalRuleEvidence", "validate_rule_span"]
