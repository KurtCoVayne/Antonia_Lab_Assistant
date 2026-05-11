"""
src/antonia/domain/utterance.py

Typed value objects that flow through pipeline stages.
All are frozen dataclasses — immutable once created.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class RawAudio:
    samples: npt.NDArray[np.float32]
    sample_rate: int
    captured_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    latency_s: float
    speech_duration_s: float = 0.0

    @property
    def is_empty(self) -> bool:
        return len(self.text.strip()) < 2


@dataclass(frozen=True)
class KnowledgeChunk:
    text: str
    source: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalContext:
    chunks: tuple[KnowledgeChunk, ...]
    query: str

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    def as_context_block(self) -> str:
        if self.is_empty:
            return ""
        lines = ["Información disponible en la base de conocimientos:"]
        for i, chunk in enumerate(self.chunks, 1):
            lines.append(f"{i}. [{chunk.source}] {chunk.text}")
        return "\n".join(lines)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    latency_s: float
    tokens_per_sec: float = 0.0
    used_retrieval: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@dataclass(frozen=True)
class SynthesisResult:
    samples: npt.NDArray[np.float32]
    sample_rate: int
    latency_s: float
    engine: str
