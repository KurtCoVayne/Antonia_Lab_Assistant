"""Unit tests for tts/sentence_buffer.py"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator

import pytest

from antonia.tts.sentence_buffer import iter_sentences


async def _stream(*tokens: str) -> AsyncGenerator[str, None]:
    for t in tokens:
        yield t


async def collect(gen: AsyncIterator[str]) -> list[str]:
    return [s async for s in gen]


@pytest.mark.asyncio
async def test_single_sentence_no_split():
    sentences = await collect(iter_sentences(_stream("Hola, ¿cómo estás?")))
    assert len(sentences) == 1
    assert "Hola" in sentences[0]


@pytest.mark.asyncio
async def test_two_sentences_split_on_period():
    tokens = ["El multímetro está en el cajón.", " Úsalo con cuidado."]
    sentences = await collect(iter_sentences(_stream(*tokens)))
    assert len(sentences) == 2
    assert "cajón" in sentences[0]
    assert "cuidado" in sentences[1]


@pytest.mark.asyncio
async def test_abbreviation_not_split():
    tokens = ["El Dr.", " García explicó el procedimiento."]
    sentences = await collect(iter_sentences(_stream(*tokens)))
    assert len(sentences) == 1
    assert "García" in sentences[0]


@pytest.mark.asyncio
async def test_decimal_not_split():
    tokens = ["La resistencia es 3.", "14 ohmios."]
    sentences = await collect(iter_sentences(_stream(*tokens)))
    assert len(sentences) == 1


@pytest.mark.asyncio
async def test_empty_stream():
    sentences = await collect(iter_sentences(_stream()))
    assert sentences == []


@pytest.mark.asyncio
async def test_aggressive_first_split_on_semicolon():
    # Buffer exceeds first_sentence_word_limit=5, hits ; before any .!?
    words = "uno dos tres cuatro cinco seis; segunda parte aquí"
    sentences = await collect(iter_sentences(_stream(words), first_sentence_word_limit=5))
    assert len(sentences) == 2


@pytest.mark.asyncio
async def test_exclamation_and_question_split():
    tokens = ["¡Cuidado! ", "El voltaje es alto."]
    sentences = await collect(iter_sentences(_stream(*tokens)))
    assert len(sentences) >= 1
    assert "Cuidado" in " ".join(sentences)
