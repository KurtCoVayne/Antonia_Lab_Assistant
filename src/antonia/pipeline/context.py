"""
src/antonia/pipeline/context.py

ConversationContext — typed conversation history with a bounded window.
Replaces the raw list[dict] historial.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class TurnRecord:
    turn: int
    user_text: str
    assistant_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    lat_stt_s: float = 0.0
    lat_llm_s: float = 0.0
    lat_tts_s: float = 0.0
    used_retrieval: bool = False


class ConversationContext:
    def __init__(self, max_messages: int = 4) -> None:
        self._max = max_messages
        self._turns: list[TurnRecord] = []

    def add_turn(self, record: TurnRecord) -> None:
        self._turns.append(record)

    def as_messages(self) -> list[dict[str, str]]:
        """Return the last N turns as a flat list of role/content dicts."""
        messages: list[dict[str, str]] = []
        for t in self._turns[-(self._max // 2):]:
            messages.append({"role": "user",      "content": t.user_text})
            messages.append({"role": "assistant", "content": t.assistant_text})
        return messages

    def clear(self) -> None:
        self._turns.clear()

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    def save(self, path: Path) -> None:
        data: list[dict[str, Any]] = [
            {
                "turn":          t.turn,
                "user":          t.user_text,
                "assistant":     t.assistant_text,
                "timestamp":     t.timestamp.isoformat(),
                "lat_stt_s":     t.lat_stt_s,
                "lat_llm_s":     t.lat_llm_s,
                "lat_tts_s":     t.lat_tts_s,
                "used_retrieval": t.used_retrieval,
            }
            for t in self._turns
        ]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
