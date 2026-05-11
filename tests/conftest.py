"""
tests/conftest.py

Shared fixtures for all tests. Uses mock backends so no hardware is required.
"""

from __future__ import annotations

import os

import pytest

# Force test profile before any antonia import
os.environ.setdefault("ANTONIA_PROFILE", "apple-silicon")


@pytest.fixture
def mock_stt():
    from antonia.stt.mock import MockSTTBackend
    return MockSTTBackend()


@pytest.fixture
def mock_llm():
    from antonia.llm.mock import MockLLMBackend
    return MockLLMBackend()


@pytest.fixture
def mock_tts():
    from antonia.tts.mock import MockTTSBackend
    return MockTTSBackend()


@pytest.fixture
def mock_kb():
    from antonia.kb.store import MockKBRetriever
    return MockKBRetriever()


@pytest.fixture
def settings():
    from antonia.config.settings import AntoniaSettings
    return AntoniaSettings(profile="apple-silicon")


@pytest.fixture
def conversation_context():
    from antonia.pipeline.context import ConversationContext
    return ConversationContext(max_messages=4)
