from app.eureka_metadata import canonical_provision_aliases, derive_tax_domain, enrich_interpretation_metadata
from app.legacy_july7.mysql_rag import local_record_to_mysql_document


def test_canonical_provisions_preserve_eureka_value_and_add_searchable_alias() -> None:
    source = "[VAT][WIS] Ustawa o podatku od towarów i usług-Dział IX-art. 86a-ust. 2-pkt 1"

    values = canonical_provision_aliases([source], tax_domain="VAT")

    assert values[0] == source
    assert "[VAT] art. 86a ust. 2 pkt 1" in values


def test_domain_mapping_recognises_eureka_ryczalt_and_inheritance_tags() -> None:
    assert derive_tax_domain(law_tags=["[ZPDOF]"]) == "PIT"
    assert derive_tax_domain(legal_provisions=["[PSD] Ustawa o podatku od spadków i darowizn-art. 1"]) == "SD"


def test_enrichment_is_idempotent_and_does_not_guess_missing_provisions() -> None:
    metadata = enrich_interpretation_metadata(
        law_tags=["[PIT]"],
        legal_provisions=["[PIT] Ustawa o PIT-art. 22-ust. 1"],
    )

    repeated = enrich_interpretation_metadata(
        tax_domain=metadata.tax_domain,
        legal_provisions=list(metadata.legal_provisions),
    )

    assert metadata.tax_domain == "PIT"
    assert "[PIT] art. 22 ust. 1" in metadata.legal_provisions
    assert repeated == metadata


def test_mysql_reindex_keeps_canonical_eureka_provisions() -> None:
    row = local_record_to_mysql_document({
        "document_id": "eureka-1",
        "source": "eureka",
        "source_type": "interpretation",
        "subject": "Odliczenie VAT od wydatku na samochód",
        "content_text": "Opis: samochód wykorzystywany w działalności.",
        "keywords": ["samochód"],
        "legal_provisions": ["[VAT] Ustawa o VAT-art. 86a-ust. 2"],
        "issues": [],
        "law_tags": ["[VAT]"],
    })

    assert row["tax_domain"] == "VAT"
    assert "[VAT] art. 86a ust. 2" in row["legal_provisions_json"]
