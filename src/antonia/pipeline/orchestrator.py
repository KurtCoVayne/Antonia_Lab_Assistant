"""
src/antonia/pipeline/orchestrator.py

AntoniaOrchestrator — the async state machine that drives one pipeline turn.
Depends only on protocols and domain types — never on concrete backend classes.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Optional

import structlog

from antonia.domain.utterance import LLMResponse, RawAudio, SynthesisResult, TranscriptionResult
from antonia.pipeline.context import ConversationContext, TurnRecord
from antonia.pipeline.relay import GPURelay
from antonia.tts.sentence_buffer import iter_sentences

log = structlog.get_logger(__name__)


class AntoniaOrchestrator:
    """
    Drives one full pipeline turn: audio → STT → KB → LLM → TTS.
    GPU relay (unload / reload) is injected and called transparently.
    """

    def __init__(
        self,
        stt: object,
        llm: object,
        tts: object,
        relay: object,
        prompt_template: object,
        kb_retriever: object | None = None,
        wav_dir: Path | None = None,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._relay = relay
        self._prompt = prompt_template
        self._kb = kb_retriever
        self._wav_dir = wav_dir

    async def run_turn(
        self,
        audio: RawAudio,
        context: ConversationContext,
        turn_number: int,
        save_wav: bool = False,
    ) -> TurnRecord:
        structlog.contextvars.bind_contextvars(turn=turn_number)
        t_start = time.time()

        # ── STT ───────────────────────────────────────────────────────────
        log.info("phase_stt_start")
        result: TranscriptionResult = await asyncio.to_thread(
            self._stt.transcribe, audio.samples  # type: ignore[union-attr]
        )
        log.info("phase_stt_done", text=result.text, latency_s=round(result.latency_s, 3))

        if result.is_empty:
            log.warning("utterance_empty")
            return TurnRecord(
                turn=turn_number,
                user_text="",
                assistant_text="",
                lat_stt_s=result.latency_s,
            )

        # ── KB Retrieval (optional) ────────────────────────────────────────
        retrieval = None
        if self._kb is not None:
            try:
                retrieval = await asyncio.to_thread(
                    self._kb.query, result.text  # type: ignore[union-attr]
                )
            except Exception as exc:
                log.warning("kb_retrieval_error", error=str(exc))

        # ── GPU Relay: Whisper out ─────────────────────────────────────────
        await self._relay.before_llm()  # type: ignore[union-attr]

        # ── LLM + TTS ─────────────────────────────────────────────────────
        messages = self._prompt.build_messages(  # type: ignore[union-attr]
            result.text, context, retrieval
        )

        if hasattr(self._llm, "ask_stream"):
            llm_resp, tts_result = await self._run_streaming(messages)
        else:
            log.info("phase_llm_start")
            llm_resp = await self._llm.ask(messages)  # type: ignore[union-attr]
            log.info("phase_llm_done", latency_s=round(llm_resp.latency_s, 3))

            tts_result = None
            if not llm_resp.is_empty:
                log.info("phase_tts_start")
                wav_name = f"turn{turn_number:02d}.wav" if save_wav else "antonia_output.wav"
                tts_result = await asyncio.to_thread(
                    self._tts.speak,  # type: ignore[union-attr]
                    llm_resp.text,
                    True,
                    save_wav,
                    wav_name,
                )
                if tts_result:
                    log.info("phase_tts_done", latency_s=round(tts_result.latency_s, 3))

        if not llm_resp.is_empty:
            context.add_turn(TurnRecord(
                turn=turn_number,
                user_text=result.text,
                assistant_text=llm_resp.text,
                lat_stt_s=result.latency_s,
                lat_llm_s=llm_resp.latency_s,
                used_retrieval=retrieval is not None and not retrieval.is_empty,
            ))

        # ── GPU Relay: Whisper back ────────────────────────────────────────
        await self._relay.after_tts()  # type: ignore[union-attr]

        lat_total = time.time() - t_start
        log.info("turn_complete", total_s=round(lat_total, 2))
        structlog.contextvars.unbind_contextvars("turn")

        return TurnRecord(
            turn=turn_number,
            user_text=result.text,
            assistant_text=llm_resp.text,
            lat_stt_s=result.latency_s,
            lat_llm_s=llm_resp.latency_s,
            lat_tts_s=tts_result.latency_s if tts_result else 0.0,
            used_retrieval=retrieval is not None and not retrieval.is_empty,
        )

    async def _run_streaming(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[LLMResponse, Optional[SynthesisResult]]:
        force_cpu = isinstance(self._relay, GPURelay)
        t0 = time.time()
        parts: list[str] = []

        log.info("phase_llm_tts_streaming_start")
        async for sentence in iter_sentences(self._llm.ask_stream(messages)):  # type: ignore[union-attr]
            parts.append(sentence)
            log.debug("sentence_ready", preview=sentence[:40])
            await asyncio.to_thread(
                self._tts.speak_sentence,  # type: ignore[union-attr]
                sentence,
                force_cpu,
            )

        full_text = " ".join(parts)
        elapsed = time.time() - t0
        words = len(full_text.split())
        llm_resp = LLMResponse(
            text=full_text,
            latency_s=elapsed,
            tokens_per_sec=words / elapsed if elapsed > 0 else 0.0,
        )
        log.info("phase_llm_tts_streaming_done", latency_s=round(elapsed, 3))
        return llm_resp, None
