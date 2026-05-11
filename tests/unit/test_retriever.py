"""
tests/unit/test_retriever.py

Unit tests for KB retriever — uses MockKBRetriever (no ChromaDB needed).
"""

from antonia.domain.utterance import RetrievalContext
from antonia.kb.store import MockKBRetriever


def test_mock_retriever_returns_empty_context():
    kb = MockKBRetriever()
    result = kb.query("¿Dónde está el PLC?")
    assert isinstance(result, RetrievalContext)
    assert result.is_empty


def test_mock_retriever_not_available():
    kb = MockKBRetriever()
    assert not kb.is_available()


def test_retrieval_context_as_context_block_when_empty():
    ctx = RetrievalContext(chunks=(), query="test")
    assert ctx.as_context_block() == ""
