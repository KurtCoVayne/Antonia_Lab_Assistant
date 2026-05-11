"""
src/antonia/domain/prompt.py

Prompt system — typed, versioned, YAML-backed.
No string literals live in Python code; all prompt content is in config/prompts/*.yaml.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    from antonia.domain.utterance import RetrievalContext
    from antonia.pipeline.context import ConversationContext


@dataclass(frozen=True)
class FewShotExample:
    user: str
    assistant: str
    category: str = ""


@dataclass(frozen=True)
class SystemPrompt:
    name: str
    version: str
    language: str
    role_description: str
    response_constraints: tuple[str, ...]
    domain_context: str
    few_shot_examples: tuple[FewShotExample, ...]

    def render(self) -> str:
        parts = [self.role_description.strip()]
        if self.domain_context.strip():
            parts.append(self.domain_context.strip())
        if self.response_constraints:
            parts.append("Reglas:")
            for i, c in enumerate(self.response_constraints, 1):
                parts.append(f"{i}. {c.strip()}")
        return "\n\n".join(parts)


class PromptTemplate:
    """
    Combines a SystemPrompt with conversation history and optional RAG context
    into the final messages list sent to the LLM.
    """

    def __init__(self, system_prompt: SystemPrompt) -> None:
        self._prompt = system_prompt

    def build_messages(
        self,
        user_input: str,
        history: "ConversationContext",
        retrieval: "RetrievalContext | None" = None,
    ) -> list[dict[str, str]]:
        system_text = self._prompt.render()

        if retrieval and not retrieval.is_empty:
            system_text = system_text + "\n\n" + retrieval.as_context_block()

        messages: list[dict[str, str]] = [{"role": "system", "content": system_text}]

        for example in self._prompt.few_shot_examples:
            messages.append({"role": "user",      "content": example.user})
            messages.append({"role": "assistant", "content": example.assistant})

        messages.extend(history.as_messages())
        messages.append({"role": "user", "content": user_input})
        return messages


class PromptRegistry:
    """
    Loads SystemPrompt definitions from config/prompts/*.yaml.
    """

    def __init__(self, prompt_dir: Path) -> None:
        self._dir = prompt_dir
        self._cache: dict[str, PromptTemplate] = {}

    def get(self, name: str) -> PromptTemplate:
        if name not in self._cache:
            self._cache[name] = self._load(name)
        return self._cache[name]

    def _load(self, name: str) -> PromptTemplate:
        path = self._dir / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")

        with path.open(encoding="utf-8") as f:
            raw: dict = yaml.safe_load(f)

        constraints = tuple(raw.get("response_constraints", []))
        examples = tuple(
            FewShotExample(
                user=ex["user"],
                assistant=ex["assistant"],
                category=ex.get("category", ""),
            )
            for ex in raw.get("few_shot_examples", [])
        )

        sp = SystemPrompt(
            name=raw["name"],
            version=str(raw.get("version", "1.0")),
            language=raw.get("language", "es"),
            role_description=re.sub(r"\s+", " ", raw.get("role_description", "")).strip(),
            response_constraints=constraints,
            domain_context=re.sub(r"\s+", " ", raw.get("domain_context", "")).strip(),
            few_shot_examples=examples,
        )
        return PromptTemplate(sp)
