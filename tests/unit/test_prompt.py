"""
tests/unit/test_prompt.py

Unit tests for the prompt system.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from antonia.domain.prompt import FewShotExample, PromptRegistry, PromptTemplate, SystemPrompt
from antonia.pipeline.context import ConversationContext


def _make_prompt() -> SystemPrompt:
    return SystemPrompt(
        name="test",
        version="1.0",
        language="es",
        role_description="Eres Antonia.",
        response_constraints=("Responde en español.",),
        domain_context="Laboratorio EAFIT.",
        few_shot_examples=(
            FewShotExample(user="Hola", assistant="Buenos días."),
        ),
    )


def test_system_prompt_render_contains_role():
    sp = _make_prompt()
    rendered = sp.render()
    assert "Eres Antonia" in rendered


def test_system_prompt_render_contains_constraints():
    sp = _make_prompt()
    rendered = sp.render()
    assert "Responde en español" in rendered


def test_prompt_template_builds_messages():
    sp = _make_prompt()
    template = PromptTemplate(sp)
    ctx = ConversationContext(max_messages=4)
    messages = template.build_messages("¿Dónde está el PLC?", ctx)
    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert roles[-1] == "user"
    assert messages[-1]["content"] == "¿Dónde está el PLC?"


def test_prompt_template_includes_history():
    from antonia.pipeline.context import TurnRecord

    sp = _make_prompt()
    template = PromptTemplate(sp)
    ctx = ConversationContext(max_messages=4)
    ctx.add_turn(TurnRecord(turn=1, user_text="Hola", assistant_text="Buenos días."))

    messages = template.build_messages("¿Y el multímetro?", ctx)
    contents = [m["content"] for m in messages]
    assert "Hola" in contents
    assert "Buenos días." in contents


def test_prompt_registry_loads_laboratory(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    (prompt_dir / "laboratory.yaml").write_text(
        """
name: laboratory
version: "1.0"
language: es
role_description: Eres Antonia.
response_constraints:
  - Responde en español.
domain_context: Laboratorio EAFIT.
few_shot_examples: []
""",
        encoding="utf-8",
    )
    registry = PromptRegistry(prompt_dir)
    template = registry.get("laboratory")
    assert isinstance(template, PromptTemplate)


def test_prompt_registry_missing_file(tmp_path):
    registry = PromptRegistry(tmp_path)
    with pytest.raises(FileNotFoundError):
        registry.get("nonexistent")
