from __future__ import annotations

from typing import Protocol

from antonia.domain.utterance import LLMResponse


class LLMBackend(Protocol):
    async def ask(
        self,
        messages: list[dict[str, str]],
    ) -> LLMResponse: ...
