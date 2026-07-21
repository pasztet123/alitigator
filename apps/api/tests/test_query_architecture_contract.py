import inspect
from pathlib import Path

from app.legal_rag_v2.document_validation import build_document_card


def test_document_card_builder_is_question_independent() -> None:
    parameters = inspect.signature(build_document_card).parameters
    assert "question" not in parameters
    assert "query_plan" not in parameters


def test_active_architecture_has_no_benchmark_specific_conditionals() -> None:
    root = Path(__file__).parents[1] / "app"
    active = [root / "legal_concepts", root / "query_understanding", root / "query_generation", root / "document_understanding", root / "legal_rag_v2" / "document_validation.py"]
    source = "\n".join(path.read_text(encoding="utf-8") for item in active for path in ([item] if item.is_file() else item.rglob("*.py")))
    forbidden = ('if "saas"', 'if "ulga sponsoringowa"', 'if "deweloper"', "build_saas_query", "build_expansion_relief_query")
    assert not any(value in source.casefold() for value in forbidden)
