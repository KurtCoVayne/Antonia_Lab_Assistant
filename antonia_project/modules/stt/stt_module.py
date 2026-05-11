"""
modules/stt/stt_module.py
Proyecto Antonia — EAFIT

Módulo STT: Whisper small GPU + DSP pipeline
Sistema de relevos GPU: unload_gpu() / reload_gpu()
"""

import os
# C-4: max_split_size_mb preserva bloques contiguos ≥512 MB para Ollama.
#      garbage_collection_threshold activa limpieza proactiva antes de OOM.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "max_split_size_mb:512,"
    "garbage_collection_threshold:0.8"
)

import gc
import time
import torch
import torchaudio
import numpy as np
from faster_whisper import WhisperModel

# ── Configuración ──────────────────────────────────────────────────────────
MODEL_SIZE = "small"
DEVICE     = "cuda"

# H-1: Expuesto como constante para ajuste sin-tocar-código en Jetson.
# float16 es más estable que int8 en JetPack 6 (comportamiento documentado
# en el hardware). Cambiar a "int8" para ~35-50% más throughput si se
# verifica estabilidad en el dispositivo.
COMPUTE_TYPE = "float16"

SR_HW      = 44100   # Sample rate nativo del mic USB
SR_TARGET  = 16000   # Sample rate requerido por Whisper
GAIN       = 3.0     # Compensación de baja sensibilidad del mic

# H-2: Resampler construido UNA VEZ al importar el módulo.
# torchaudio usa NEON/SVE en ARM64 (JetPack toolchain) → 3-6× más rápido
# que librosa/soxr que no tiene vectorización ARM en el wheel de PyPI.
# sinc_interp_hann: 3× más rápido que kaiser, sin diferencia audible
# para la etapa de entrada de Whisper.
_resampler = torchaudio.transforms.Resample(
    orig_freq=SR_HW,
    new_freq=SR_TARGET,
    resampling_method="sinc_interp_hann",
)

# Umbrales para unload_gpu() polling
_VRAM_TARGET_MB  = 50
_POLL_INTERVAL_S = 0.05
_MAX_POLLS       = 20   # 20 × 50ms = 1.0s máximo


class AntoniaSTT:
    """
    Whisper con gestión explícita de VRAM para el sistema de relevos.

    Ciclo de uso desde el pipeline:
        texto, lat = stt.transcribe(audio_44k)
        stt.unload_gpu()          # libera VRAM antes de Ollama
        ...llamada a LLM...
        ...TTS reproduce en CPU...
        stt.reload_gpu()          # recarga Whisper después de TTS
    """

    def __init__(self):
        self._loaded = False
        self.whisper = None
        self._load()

    # ── Ciclo de vida en VRAM ──────────────────────────────────────────────

    def _load(self) -> None:
        print(f"[STT]  Cargando Whisper '{MODEL_SIZE}' ({DEVICE}, {COMPUTE_TYPE})...")
        t0 = time.time()
        self.whisper = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            device_index=0,
            num_workers=1,
            cpu_threads=4,
        )
        self._loaded = True
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"[STT]  ✅ Listo en {time.time() - t0:.2f}s  (VRAM: {mem:.0f} MB)")

    def unload_gpu(self) -> None:
        """
        Descarga Whisper de GPU.
        Llama ANTES de enviar la petición a Ollama.
        """
        if not self._loaded:
            return

        del self.whisper
        self.whisper  = None
        self._loaded  = False

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # C-4: Polling determinista en lugar de sleep fijo.
        # En Tegra (memoria unificada), la consolidación física es no-determinista.
        # Máximo 1.0s (20 × 50ms); sale antes si la VRAM ya está limpia.
        for _ in range(_MAX_POLLS):
            if torch.cuda.memory_reserved() / 1024 ** 2 < _VRAM_TARGET_MB:
                break
            torch.cuda.empty_cache()
            time.sleep(_POLL_INTERVAL_S)

        mem    = torch.cuda.memory_allocated() / 1024 ** 2
        estado = "✅" if mem < _VRAM_TARGET_MB else "⚠ ALTA"
        print(f"[STT]  🔄 Whisper descargado. VRAM residual: {mem:.0f} MB {estado}")

    def reload_gpu(self) -> None:
        """
        Recarga Whisper en GPU.
        Llama DESPUÉS de que TTS termine de reproducir.
        """
        if self._loaded:
            return
        time.sleep(0.3)   # Margen para que KEEP_ALIVE expire en Ollama
        self._load()

    # ── DSP ────────────────────────────────────────────────────────────────

    @staticmethod
    def prepare_audio(
        raw: np.ndarray,
        gain: float = GAIN,
    ) -> np.ndarray:
        """
        Pipeline DSP: ganancia → resampleo 44100→16000 Hz (torchaudio NEON)
                    → pre-énfasis → normalización de pico.
        """
        audio = np.clip(raw * gain, -1.0, 1.0)
        # H-2: torchaudio en lugar de librosa para resampleo vectorizado en ARM64
        t     = torch.from_numpy(audio).float().unsqueeze(0)
        audio = _resampler(t).squeeze(0).numpy()
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])  # pre-énfasis
        peak  = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95
        return audio

    # ── Transcripción ──────────────────────────────────────────────────────

    def transcribe(
        self,
        audio: np.ndarray,
        already_16k: bool = False,
    ) -> tuple[str, float]:
        """
        Transcribe audio del micrófono.

        Args:
            audio:       Array float32. Si viene del mic → 44100 Hz.
                         Si ya fue resampleado → pasar already_16k=True.
            already_16k: True para saltar el DSP.

        Returns:
            (texto, latencia_segundos)
        """
        if not self._loaded:
            raise RuntimeError(
                "[STT] Whisper no está cargado. Llama reload_gpu() primero."
            )

        if not already_16k:
            audio = self.prepare_audio(audio)

        if np.max(np.abs(audio)) < 0.04:
            return "", 0.0

        t0 = time.time()
        segments, _ = self.whisper.transcribe(
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
            initial_prompt=(
                "Transcripción en el Laboratorio de Control Digital, "
                "Universidad EAFIT, Medellín. Términos posibles: "
                "PLC, osciloscopio, multímetro, EAFIT, Siemens."
            ),
        )
        texto = " ".join(s.text.strip() for s in segments)
        return texto, time.time() - t0
