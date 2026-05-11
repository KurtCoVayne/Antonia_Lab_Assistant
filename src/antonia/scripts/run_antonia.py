"""
src/antonia/scripts/run_antonia.py

Entry point: antonia-run
Runs N_TURNS of the full STT → LLM → TTS pipeline with manual recording.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path

import httpx
import structlog

N_TURNS = 3


async def _main() -> None:
    # Import after configure_logging so all sub-modules get the right config
    from antonia.config.settings import settings
    from antonia.domain.utterance import RawAudio
    from antonia.factory import build_kb, build_llm, build_prompt_template, build_relay, build_stt, build_tts
    from antonia.infra.health import check_ollama
    from antonia.infra.logging import configure_logging
    from antonia.pipeline.context import ConversationContext
    from antonia.pipeline.orchestrator import AntoniaOrchestrator

    configure_logging(level=settings.logging.level, fmt=settings.logging.format)
    log = structlog.get_logger(__name__)

    log.info("startup", profile=settings.profile, timestamp=datetime.now().isoformat())

    async with httpx.AsyncClient(timeout=60.0) as client:
        ok = await check_ollama(settings.llm.base_url, settings.llm.model)
        if not ok:
            log.error("ollama_not_ready")
            return

        log.info("loading_stt")
        stt = await asyncio.to_thread(build_stt, settings)
        tts = build_tts(settings)
        llm = build_llm(settings, client)
        relay = build_relay(settings, stt)
        kb = build_kb(settings)
        prompt = build_prompt_template(settings)

        from antonia.audio.capture import AudioCapture
        from antonia.audio.listener import SmartListener
        from antonia.audio.vad import SileroVAD

        capture = AudioCapture(
            sample_rate=settings.audio.sample_rate_hw,
            chunk_samples=settings.audio.chunk_samples,
            device=settings.audio.device_index,
        )
        vad = await asyncio.to_thread(SileroVAD)
        listener = SmartListener(
            capture=capture,
            vad=vad,
            sample_rate_hw=settings.audio.sample_rate_hw,
            vad_threshold=settings.vad_threshold,
            silence_windows=settings.silence_windows,
            min_speech_windows=settings.min_speech_windows,
        )

        orchestrator = AntoniaOrchestrator(
            stt=stt,
            llm=llm,
            tts=tts,
            relay=relay,
            prompt_template=prompt,
            kb_retriever=kb,
            wav_dir=settings.base_dir / "tests" / "pipeline_runs",
        )

        context = ConversationContext(max_messages=settings.history_max_messages)

        import numpy as np
        from datetime import datetime as dt

        capture.start()
        try:
            for turn in range(1, N_TURNS + 1):
                log.info("turn_start", turn=turn, total=N_TURNS)
                log.info("listening_for_speech")
                raw = await asyncio.to_thread(listener.listen_until_silence)
                audio = RawAudio(
                    samples=raw.astype(np.float32),
                    sample_rate=settings.audio.sample_rate_hw,
                    captured_at=dt.now(),
                )
                record = await orchestrator.run_turn(audio, context, turn, save_wav=True)
                log.info(
                    "turn_done",
                    user=record.user_text,
                    assistant=record.assistant_text,
                    lat_stt=round(record.lat_stt_s, 2),
                    lat_llm=round(record.lat_llm_s, 2),
                    lat_tts=round(record.lat_tts_s, 2),
                )
        finally:
            capture.stop()

    log.info("session_complete", turns=N_TURNS)


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
