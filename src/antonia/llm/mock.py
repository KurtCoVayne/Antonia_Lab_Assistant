from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator

from antonia.domain.utterance import LLMResponse


class MockLLMBackend:
    def __init__(self, response: str = "Los multímetros están en el cajón superior.") -> None:
        self._response = response

    async def ask(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(text=self._response, latency_s=0.01, tokens_per_sec=100.0)

    async def ask_stream(self, messages: list[dict[str, str]]) -> AsyncGenerator[str, None]:
        for word in self._response.split():
            yield word + " "
            await asyncio.sleep(0.001)
