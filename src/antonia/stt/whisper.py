"""
src/antonia/stt/whisper.py

WhisperBackend — faster-whisper / CTranslate2 implementation.
Supports both CUDA (Jetson) and CPU (Mac M4 / cpu-only) modes.
GPU relay (unload_gpu / reload_gpu) is a no-op when device=cpu.
"""

from __future__ import annotations

import os
import time

import numpy as np
import numpy.typing as npt
import structlog

from antonia.audio.dsp import prepare_for_whisper
from antonia.config.settings import STTConfig
from antonia.domain.utterance import TranscriptionResult

# C-4: CUDA allocator settings must be set before torch is imported.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "expandable_segments:True,max_split_size_mb:512,garbage_collection_threshold:0.8",
)

log = structlog.get_logger(__name__)

_SILENCE_THRESHOLD = 0.04
_INITIAL_PROMPT = (
    "Transcripción en el Laboratorio de Control Digital, Universidad EAFIT, Medellín. "
    "Términos posibles: PLC, osciloscopio, multímetro, EAFIT, Siemens."
)


class WhisperBackend:
    """
    Whisper STT with explicit GPU lifecycle for the relay system.

    On CPU profiles, unload_gpu() and reload_gpu() are no-ops.
    """

    def __init__(self, config: STTConfig, sr_hw: int = 44100, gain: float = 3.0) -> None:
        self._config = config
        self._sr_hw = sr_hw
        self._gain = gain
        self._loaded = False
        self._model: object = None
        self._load()

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        from faster_whisper import WhisperModel

        log.info("stt_loading", model=self._config.model_size, device=self._config.device)
        t0 = time.time()
        self._model = WhisperModel(
            self._config.model_size,
            device=self._config.device,
            compute_type=self._config.compute_type,
            device_index=self._config.device_index,
            num_workers=self._config.num_workers,
            cpu_threads=self._config.cpu_threads,
        )
        self._loaded = True
        elapsed = time.time() - t0
        vram_mb = self._vram_allocated()
        log.info("stt_ready", elapsed_s=round(elapsed, 2), vram_mb=round(vram_mb, 0))

    def unload_gpu(self) -> None:
        if not self._loaded or self._config.device == "cpu":
            return
        import gc
        import torch

        del self._model
        self._model = None
        self._loaded = False
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        log.info("stt_unloaded", vram_mb=round(self._vram_allocated(), 0))

    def reload_gpu(self) -> None:
        if self._loaded or self._config.device == "cpu":
            return
        time.sleep(0.3)
        self._load()

    # ── Transcription ──────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: npt.NDArray[np.float32],
        already_16k: bool = False,
    ) -> TranscriptionResult:
        if not self._loaded:
            raise RuntimeError("WhisperBackend not loaded. Call reload_gpu() first.")

        if not already_16k:
            audio = prepare_for_whisper(audio, self._sr_hw, gain=self._gain)

        if np.max(np.abs(audio)) < _SILENCE_THRESHOLD:
            return TranscriptionResult(text="", latency_s=0.0)

        t0 = time.time()
        segments, _ = self._model.transcribe(  # type: ignore[union-attr]
            audio,
            language="es",
            beam_size=5,
            best_of=5,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=400,
                speech_pad_ms=250,
                threshold=0.15,
            ),
            temperature=0.0,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=_INITIAL_PROMPT,
        )
        text = " ".join(s.text.strip() for s in segments)
        latency = time.time() - t0
        log.debug("stt_transcribed", latency_s=round(latency, 3), text_len=len(text))
        return TranscriptionResult(text=text, latency_s=latency)

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _vram_allocated() -> float:
        try:
            import torch
            return torch.cuda.memory_allocated() / 1024**2
        except ImportError:
            return 0.0
