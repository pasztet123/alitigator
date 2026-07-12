from app.legal_rag_v2.retrieval import TransparentLegalReranker
from app.legal_research.models import LegalRerankResult

LegalReranker = TransparentLegalReranker

__all__ = ["LegalRerankResult", "LegalReranker", "TransparentLegalReranker"]
