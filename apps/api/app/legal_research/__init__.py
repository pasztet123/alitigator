"""Model → RAG → Model legal research pipeline.

The package is the stable public API for the new architecture.  Mature,
provider-neutral components are adapted from ``legal_rag_v2`` while migration
continues; legacy query rules are never imported on the normal path.
"""

from .config import LegalResearchConfig
from .models import *
from .pipeline import ModelRagModelPipeline, create_default_pipeline

__all__ = ["LegalResearchConfig", "ModelRagModelPipeline", "create_default_pipeline"]
