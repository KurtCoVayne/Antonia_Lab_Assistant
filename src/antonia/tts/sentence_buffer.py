"""
src/antonia/tts/sentence_buffer.py

Converts a stream of LLM tokens into complete sentences for incremental TTS.
"""

from __future__ import annotations

import re
from collections.abc import AsyncGenerator, AsyncIterator

# Split on [.!?] followed by whitespace+uppercase or end of string.
# We use a post-match check to skip abbreviations and decimals rather than
# variable-width lookbehinds (which Python re does not support).
_TERMINAL_RE = re.compile(
    r"(?<!\d)"          # not preceded by a digit (decimal left guard)
    r"[.!?]"
    r"(?!\d)"           # not followed by a digit (decimal right guard)
    r"(?=\s+[A-ZÁÉÍÓÚÜÑ]|\s*$)",
    re.UNICODE,
)

# Abbreviations that should not trigger a sentence split
_ABBREV = frozenset([
    "Dr", "Sr", "Sra", "Srta", "etc", "vs", "Fig", "No", "Núm", "pág",
    "aprox", "Ing", "Dra", "Prof", "Lic",
])

# Aggressive first-sentence split: also break on ; or : when buffer is long
_EARLY_SPLIT_RE = re.compile(r"[;:]")


def _word_count(text: str) -> int:
    return len(text.split())


def _is_abbreviation(buf: str, match_start: int) -> bool:
    """True if the word before the terminal punctuation is a known abbreviation."""
    before = buf[:match_start].rstrip()
    word = before.split()[-1] if before.split() else ""
    return word in _ABBREV


def _find_split(buf: str, first_emitted: bool, first_sentence_word_limit: int) -> int | None:
    """Return the index after the split character, or None if no split found."""
    for m in _TERMINAL_RE.finditer(buf):
        if not _is_abbreviation(buf, m.start()):
            return m.end()
    if not first_emitted and _word_count(buf) >= first_sentence_word_limit:
        m2 = _EARLY_SPLIT_RE.search(buf)
        if m2:
            return m2.end()
    return None


async def iter_sentences(
    token_stream: AsyncIterator[str],
    first_sentence_word_limit: int = 15,
) -> AsyncGenerator[str, None]:
    """Yield complete sentences from a token stream."""
    buf = ""
    first_emitted = False

    async for token in token_stream:
        buf += token
        while True:
            idx = _find_split(buf, first_emitted, first_sentence_word_limit)
            if idx is None:
                break
            sentence = buf[:idx].strip()
            buf = buf[idx:]
            if sentence:
                yield sentence
                first_emitted = True

    remainder = buf.strip()
    if remainder:
        yield remainder
