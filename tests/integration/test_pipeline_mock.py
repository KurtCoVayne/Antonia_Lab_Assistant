"""
tests/integration/test_pipeline_mock.py

Full pipeline integration test using mock backends — no hardware required.
"""

from __future__ import annotations

import numpy as np
import pytest

from antonia.domain.utterance import RawAudio
from antonia.pipeline.context import ConversationContext
from antonia.pipeline.orchestrator import AntoniaOrchestrator
from antonia.pipeline.relay import NullRelay


@pytest.mark.asyncio
async def test_full_pipeline_mock(mock_stt, mock_llm, mock_tts, mock_kb, settings):
    from antonia.domain.prompt import FewShotExample, PromptTemplate, SystemPrompt
    from datetime import datetime

    sp = SystemPrompt(
        name="test",
        version="1.0",
        language="es",
        role_description="Eres Antonia.",
        response_constraints=("Responde en español.",),
        domain_context="Lab EAFIT.",
        few_shot_examples=(),
    )
    prompt = PromptTemplate(sp)
    relay = NullRelay()
    ctx = ConversationContext(max_messages=4)

    orchestrator = AntoniaOrchestrator(
        stt=mock_stt,
        llm=mock_llm,
        tts=mock_tts,
        relay=relay,
        prompt_template=prompt,
        kb_retriever=mock_kb,
    )

    audio = RawAudio(
        samples=np.zeros(44100, dtype=np.float32),
        sample_rate=44100,
        captured_at=datetime.now(),
    )

    record = await orchestrator.run_turn(audio, ctx, turn_number=1, save_wav=False)

    assert record.turn == 1
    assert record.user_text == "¿Dónde están los multímetros?"
    assert "multímetros" in record.assistant_text.lower() or record.assistant_text


@pytest.mark.asyncio
async def test_empty_stt_skips_llm(mock_llm, mock_tts, mock_kb, settings):
    from antonia.stt.mock import MockSTTBackend
    from antonia.domain.prompt import FewShotExample, PromptTemplate, SystemPrompt
    from datetime import datetime

    stt = MockSTTBackend(response="")  # empty transcription
    sp = SystemPrompt(
        name="test",
        version="1.0",
        language="es",
        role_description="Eres Antonia.",
        response_constraints=(),
        domain_context="",
        few_shot_examples=(),
    )
    prompt = PromptTemplate(sp)
    relay = NullRelay()
    ctx = ConversationContext(max_messages=4)

    orchestrator = AntoniaOrchestrator(
        stt=stt,
        llm=mock_llm,
        tts=mock_tts,
        relay=relay,
        prompt_template=prompt,
    )

    audio = RawAudio(
        samples=np.zeros(44100, dtype=np.float32),
        sample_rate=44100,
        captured_at=datetime.now(),
    )
    record = await orchestrator.run_turn(audio, ctx, turn_number=1)
    assert record.user_text == ""
    assert record.assistant_text == ""
    assert ctx.turn_count == 0
