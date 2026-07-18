from __future__ import annotations

import unittest
import tempfile
from dataclasses import replace
from pathlib import Path

from app.legal_rag_v2.backends import CorpusFtsBackend
from app.legal_rag_v2.cash_payments import enrich_cash_payment_cost_plan
from app.legal_rag_v2.retrieval import LegalRetriever, RetrievalConfig
from app.legal_rag_v2.schemas import Clarification, LegalIssue, LegalResearchPlan, ResearchIntent
from app.rag import get_connection, get_rag_config, index_record


QUESTION = """
Prowadzę jednoosobową działalność gospodarczą opodatkowaną PIT. Kupiłem od
innego przedsiębiorcy sprzęt za 18 000 zł w jednej umowie i fakturze. Zapłaciłem
gotówką w dwóch ratach po 9 000 zł. Czy zachowam koszt, jak skorygować koszt i
czy zwrot gotówki oraz ponowny przelew może naprawić rozliczenie?
"""


def generic_plan() -> LegalResearchPlan:
    return LegalResearchPlan(
        user_query=QUESTION,
        intent=ResearchIntent(mode="mixed_analysis"),
        issues=[
            LegalIssue(
                issue_id="pit_cost_deductibility",
                label="PIT: koszt uzyskania przychodów i ustawowe wyłączenia",
                tax_domains=["PIT"],
                legal_mechanism="pit_cost_deductibility",
            )
        ],
        clarification=Clarification(),
        confidence=0.5,
    )


class CashPaymentCostEvidenceTests(unittest.TestCase):
    def test_replaces_generic_cost_rule_with_controlling_payment_bundle(self) -> None:
        enriched = enrich_cash_payment_cost_plan(generic_plan(), QUESTION)

        self.assertEqual(1, len(enriched.issues))
        issue = enriched.issues[0]
        self.assertEqual("pit_cash_payment_cost_exclusion", issue.issue_id)
        self.assertEqual("cash_payment_cost_exclusion", issue.legal_mechanism)
        primary_queries = {
            query.query for query in issue.query_families if query.lane == "primary_law"
        }
        self.assertEqual(
            {
                "PP art. 19",
                "PIT art. 22p ust. 1",
                "PIT art. 22p ust. 2",
                "PIT art. 22p ust. 3",
            },
            primary_queries,
        )
        self.assertNotIn("PIT art. 23 ust. 1", primary_queries)

    def test_replaces_unscoped_planner_fallback_instead_of_adding_noise_issue(self) -> None:
        fallback_plan = generic_plan().model_copy(
            update={
                "issues": [
                    LegalIssue(
                        issue_id="pit_general_tax_issue",
                        label="PIT: ogólna analiza podatkowa",
                        tax_domains=["PIT"],
                        legal_mechanism="general_tax_analysis",
                    )
                ]
            }
        )

        enriched = enrich_cash_payment_cost_plan(fallback_plan, QUESTION)

        self.assertEqual(1, len(enriched.issues))
        self.assertEqual(
            "cash_payment_cost_exclusion", enriched.issues[0].legal_mechanism
        )

    def test_authority_queries_cover_corrective_payment_without_document_ids(self) -> None:
        issue = enrich_cash_payment_cost_plan(generic_plan(), QUESTION).issues[0]
        authority_queries = {
            query.query for query in issue.query_families if query.lane == "authority"
        }

        joined = " ".join(authority_queries).casefold()
        self.assertIn("zwrot", joined)
        self.assertIn("ponowne uregulowanie", joined)
        self.assertIn("dodatkowy przelew bez zwrotu", joined)
        self.assertNotIn("0112-kdil", joined)
        self.assertNotIn("0115-kdit", joined)

    def test_classifier_handles_other_cash_payment_wordings_and_cit(self) -> None:
        questions = (
            "Czy wydatek w CIT jest kosztem, gdy transakcję 20 000 zł opłacono gotówką?",
            "Spółka podzieliła płatność gotówkową na raty poniżej limitu płatności.",
            "Czy zwrot gotówki i ponowne uregulowanie przelewem koryguje koszt podatkowy?",
        )
        for question in questions:
            with self.subTest(question=question):
                enriched = enrich_cash_payment_cost_plan(
                    generic_plan().model_copy(update={"user_query": question}),
                    question,
                )
                cash_issue = next(
                    issue
                    for issue in enriched.issues
                    if issue.legal_mechanism == "cash_payment_cost_exclusion"
                )
                self.assertIn("PP art. 19", {q.query for q in cash_issue.query_families})


class CashPaymentCorpusRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_statutory_bundle_and_corrective_authorities_are_retrieved(self) -> None:
        records = [
            {
                "document_id": "business-law-art-19",
                "source_type": "statute",
                "source_subtype": "consolidated_text",
                "act_title": "Prawo przedsiębiorców",
                "subject": "Prawo przedsiębiorców - art. 19",
                "legal_provisions": ["art. 19"],
                "issues": ["payment_channel"],
                "law_tags": ["PP", "Prawo przedsiębiorców"],
                "content_text": (
                    "Art. 19. Jednorazowa wartość transakcji bez względu na liczbę "
                    "wynikających z niej płatności; rachunek płatniczy przedsiębiorcy."
                ),
                "pre_chunked": True,
            },
            {
                "document_id": "pit-art-22p",
                "source_type": "statute",
                "source_subtype": "consolidated_text",
                "act_title": "Ustawa o podatku dochodowym od osób fizycznych",
                "subject": "PIT - art. 22p",
                "legal_provisions": ["art. 22p"],
                "issues": ["pit"],
                "law_tags": ["PIT"],
                "content_text": "Art. 22p. Wyłączenie kosztu i korekta płatności.",
                "pre_chunked": True,
                "provision_units": [
                    {
                        "citation": f"art. 22p ust. {section}",
                        "text": text,
                        "article": "22p",
                        "section": str(section),
                        "unit_type": "section",
                    }
                    for section, text in (
                        (1, "Płatność bez pośrednictwa rachunku płatniczego."),
                        (2, "Zmniejszenie kosztów albo zwiększenie przychodów."),
                        (3, "Odpowiednie stosowanie do środków trwałych."),
                    )
                ],
            },
            {
                "document_id": "interpretation-corrective-cash-2025",
                "source_type": "interpretation",
                "authority": "Dyrektor Krajowej Informacji Skarbowej",
                "subject": "Zwrot gotówki i ponowna płatność przez rachunek",
                "signature": "0112-KDIL2-2.4011.345.2025.1.MC",
                "published_date": "2025-06-11",
                "legal_provisions": ["art. 22p", "art. 19"],
                "issues": ["PIT", "koszty uzyskania przychodów"],
                "law_tags": ["PIT"],
                "content_text": (
                    "Kontrahent zwrócił gotówkę, a podatnik ponownie uregulował "
                    "zobowiązanie przelewem przez rachunek płatniczy. Organ uznał "
                    "korektę formy płatności i koszt na podstawie art. 22p oraz art. 19."
                ),
            },
            {
                "document_id": "interpretation-corrective-payment-2021",
                "source_type": "interpretation",
                "authority": "Dyrektor Krajowej Informacji Skarbowej",
                "subject": "Anulowanie pierwotnej płatności i ponowne uregulowanie",
                "signature": "0115-KDIT1.4011.692.1.2021.MT",
                "published_date": "2021-12-06",
                "legal_provisions": ["art. 22p"],
                "issues": ["PIT", "koszty uzyskania przychodów"],
                "law_tags": ["PIT"],
                "content_text": (
                    "Planowane anulowanie pierwotnej płatności oraz ponowne "
                    "uregulowanie należności przelewem na rachunek kontrahenta. "
                    "Analiza skutków w kosztach na podstawie art. 22p."
                ),
            },
            {
                "document_id": "interpretation-generic-art-22p-neighbor",
                "source_type": "interpretation",
                "authority": "Dyrektor Krajowej Informacji Skarbowej",
                "subject": "Ogólne dokumentowanie kosztów działalności",
                "signature": "GENERIC-ART-22P-NEIGHBOR",
                "published_date": "2026-01-01",
                "legal_provisions": ["art. 22p"],
                "issues": ["PIT", "dokumentowanie kosztów"],
                "law_tags": ["PIT"],
                "content_text": (
                    "Ogólna analiza dokumentowania kosztów działalności i ewidencji. "
                    "Przytoczono art. 22p bez stanu faktycznego dotyczącego formy zapłaty."
                ),
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "cash-payment-rag.sqlite3"
            config = replace(get_rag_config(), db_path=database)
            connection = get_connection(database)
            try:
                for record in records:
                    index_record(connection, record, config)
                connection.commit()
            finally:
                connection.close()

            plan = enrich_cash_payment_cost_plan(generic_plan(), QUESTION)
            result = await LegalRetriever(
                CorpusFtsBackend(backend="sqlite", sqlite_path=database),
                config=RetrievalConfig(
                    lexical_limit_per_query=30,
                    selected_limit_per_issue=20,
                ),
            ).retrieve(plan)

        primary_references = {
            reference
            for lane in result.primary_law
            for candidate in lane.candidates
            for reference in candidate.metadata.get("legal_provisions", [])
        }
        authority_signatures = [
            candidate.metadata.get("signature")
            for lane in result.authorities
            for candidate in lane.candidates[:20]
        ]
        self.assertIn("art. 19", primary_references)
        self.assertTrue(
            {"art. 22p ust. 1", "art. 22p ust. 2", "art. 22p ust. 3"}.issubset(
                primary_references
            )
        )
        self.assertIn("0112-KDIL2-2.4011.345.2025.1.MC", authority_signatures)
        self.assertIn("0115-KDIT1.4011.692.1.2021.MT", authority_signatures)
        self.assertNotIn("GENERIC-ART-22P-NEIGHBOR", authority_signatures)


if __name__ == "__main__":
    unittest.main()
