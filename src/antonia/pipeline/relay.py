"""
src/antonia/pipeline/relay.py

GPURelay — coordinates STT unload before LLM and reload after TTS.
NullRelay is used on non-GPU profiles (Mac M4, cpu-only).
"""

from __future__ import annotations

import asyncio
import structlog

log = structlog.get_logger(__name__)


class GPURelay:
    def __init__(self, stt: object, memory: object) -> None:
        self._stt = stt
        self._mem = memory

    async def before_llm(self) -> None:
        log.info("relay_unloading_whisper")
        await asyncio.gather(
            asyncio.to_thread(self._stt.unload_gpu),  # type: ignore[union-attr]
            self._mem.drop_os_cache(),  # type: ignore[union-attr]
        )

    async def after_tts(self) -> None:
        log.info("relay_reloading_whisper")
        await asyncio.to_thread(self._stt.reload_gpu)  # type: ignore[union-attr]


class NullRelay:
    async def before_llm(self) -> None:
        pass

    async def after_tts(self) -> None:
        pass
