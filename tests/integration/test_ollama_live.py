"""
tests/integration/test_ollama_live.py

Live integration test — skipped unless ANTONIA_LIVE_TESTS=1.
Requires Ollama to be running with the configured model loaded.
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ANTONIA_LIVE_TESTS") != "1",
    reason="Set ANTONIA_LIVE_TESTS=1 to run live tests",
)


@pytest.mark.asyncio
async def test_ollama_responds(settings):
    from antonia.llm.ollama import OllamaBackend

    async with httpx.AsyncClient(timeout=30.0) as client:
        backend = OllamaBackend(config=settings.llm, client=client)
        messages = [
            {"role": "system",  "content": "Responde en una sola oración."},
            {"role": "user",    "content": "¿Cuánto es dos más dos?"},
        ]
        resp = await backend.ask(messages)
        assert not resp.is_empty
        assert resp.latency_s > 0
