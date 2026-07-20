from app.main import build_july7_interpretations_reply
from app.legacy_july7.rag import RagChunk


def test_july7_interpretation_reply_keeps_full_document_without_truncation() -> None:
    full_text = "Początek.\n\n" + ("Pełna treść interpretacji. " * 80) + "\n\nKoniec."
    chunk = RagChunk(
        chunk_id="interpretation-1:0",
        document_id="interpretation-1",
        chunk_index=0,
        score=99.0,
        chunk_text=full_text,
        subject="Implanty zębowe",
        signature="0112-KDIL2-2.4011.8.2026.3.MM",
        published_date="2026-03-04",
        source_url="https://example.test/interpretation-1",
        category="Interpretacja indywidualna",
    )

    reply = build_july7_interpretations_reply([chunk])

    assert full_text in reply
    assert "### 1. 0112-KDIL2-2.4011.8.2026.3.MM" in reply
    assert "**Pełna treść interpretacji**" in reply
