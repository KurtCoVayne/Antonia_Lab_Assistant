"""
src/antonia/tts/backend.py

TTSBackend — top-level TTS facade that wraps preprocessor + engine selection.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import sounddevice as sd
import soundfile as sf
import structlog

from antonia.domain.utterance import SynthesisResult
from antonia.tts.preprocessor import TextPreprocessor

log = structlog.get_logger(__name__)


class TTSBackend:
    """
    Combines TextPreprocessor + Kokoro (primary) + Piper (fallback).
    Engines are injected; TTSBackend itself is engine-agnostic.
    """

    def __init__(
        self,
        preprocessor: TextPreprocessor,
        kokoro: object | None = None,
        piper: object | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self._pre = preprocessor
        self._kokoro = kokoro
        self._piper = piper
        self._output_dir = output_dir

        if kokoro is None and piper is None:
            raise RuntimeError("TTSBackend: at least one engine must be provided")

        engine_name = "kokoro" if kokoro else "piper"
        log.info("tts_ready", engine=engine_name)

    def speak(
        self,
        text: str,
        play_audio: bool = True,
        save_wav: bool = False,
        wav_filename: str = "antonia_output.wav",
    ) -> Optional[SynthesisResult]:
        if not text.strip():
            log.warning("tts_empty_text")
            return None

        t0 = time.time()
        processed = self._pre.process(text)
        if not processed:
            return None

        samples, sr, engine = self._synthesize(processed)
        if samples is None:
            log.error("tts_synthesis_failed")
            return None

        latency = time.time() - t0
        log.debug("tts_synthesized", engine=engine, latency_s=round(latency, 3))

        if play_audio:
            try:
                sd.play(samples, sr)
                sd.wait()
            except Exception as exc:
                log.warning("tts_playback_error", error=str(exc))

        if save_wav and self._output_dir:
            path = self._output_dir / wav_filename
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(path), samples, sr)
                log.debug("tts_wav_saved", path=str(path))
            except Exception as exc:
                log.warning("tts_wav_save_error", error=str(exc))

        return SynthesisResult(samples=samples, sample_rate=sr, latency_s=latency, engine=engine)

    def speak_sentence(
        self,
        text: str,
        force_cpu: bool = False,
    ) -> Optional[SynthesisResult]:
        if not text.strip():
            return None

        t0 = time.time()
        processed = self._pre.process(text)
        if not processed:
            return None

        samples, sr, engine = self._synthesize(processed, force_cpu=force_cpu)
        if samples is None:
            log.error("tts_synthesis_failed")
            return None

        latency = time.time() - t0
        try:
            sd.play(samples, sr)
            sd.wait()
        except Exception as exc:
            log.warning("tts_playback_error", error=str(exc))

        return SynthesisResult(samples=samples, sample_rate=sr, latency_s=latency, engine=engine)

    def _synthesize(
        self, text: str, force_cpu: bool = False
    ) -> tuple[Optional[npt.NDArray[np.float32]], int, str]:
        if self._kokoro is not None:
            try:
                samples, sr = self._kokoro.synthesize(text, force_cpu=force_cpu)  # type: ignore[union-attr]
                return samples, sr, "kokoro"
            except Exception as exc:
                log.warning("kokoro_failed", error=str(exc))

        if self._piper is not None:
            try:
                samples, sr = self._piper.synthesize(text)  # type: ignore[union-attr]
                return samples, sr, "piper"
            except Exception as exc:
                log.error("piper_failed", error=str(exc))

        return None, 0, "none"
