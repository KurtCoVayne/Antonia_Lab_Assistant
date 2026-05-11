from __future__ import annotations

from antonia.domain.utterance import LLMResponse


class MockLLMBackend:
    def __init__(self, response: str = "Los multímetros están en el cajón superior.") -> None:
        self._response = response

    async def ask(self, messages: list[dict[str, str]]) -> LLMResponse:
        return LLMResponse(text=self._response, latency_s=0.01, tokens_per_sec=100.0)
