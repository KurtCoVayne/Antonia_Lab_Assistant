"""
src/antonia/domain/knowledge.py

Knowledge base query types.
KnowledgeChunk and RetrievalContext are defined in utterance.py to avoid circular imports.
This module re-exports them for callers that prefer the domain.knowledge namespace.
"""

from antonia.domain.utterance import KnowledgeChunk, RetrievalContext

__all__ = ["KnowledgeChunk", "RetrievalContext"]
