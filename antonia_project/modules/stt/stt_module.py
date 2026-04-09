"""
modules/stt/stt_module.py
Proyecto Antonia — EAFIT

Módulo STT: Whisper small GPU float16 + DSP pipeline
Sistema de relevos GPU: unload_gpu() / reload_gpu()
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import time
import torch
import numpy as np
import librosa
from faster_whisper import WhisperModel

# ── Configuración ──────────────────────────────────────────────────────────
MODEL_SIZE  = "small"
DEVICE      = "cuda"
COMPUTE     = "float16"   # float16 nativo Tegra — más estable que int8 en JetPack 6
SR_HW       = 44100       # Sample rate nativo del mic USB
SR_TARGET   = 16000       # Sample rate requerido por Whisper
GAIN        = 3.0         # Compensación de baja sensibilidad del mic


class AntoniaSTT:
    """
    Whisper con gestión explícita de VRAM para el sistema de relevos.

    Ciclo de uso desde el pipeline:
        texto, lat = stt.transcribe(audio_44k)
        stt.unload_gpu()          # libera VRAM antes de Ollama
        ...llamada a LLM...
        ...TTS reproduce en CPU... # no hay conflicto de GPU aquí
        stt.reload_gpu()          # recarga Whisper después de TTS
    """

    def __init__(self):
        self._loaded = False
        self.whisper = None
        self._load()

    # ── Ciclo de vida en VRAM ──────────────────────────────────────────────

    def _load(self) -> None:
        print(f"[STT]  Cargando Whisper '{MODEL_SIZE}' ({DEVICE}, {COMPUTE})...")
        t0 = time.time()
        self.whisper = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE,
            device_index=0,   # Ruteo explícito al bus GPU principal
            num_workers=1,    # Evita clonación de tensores en RAM en la carga
            cpu_threads=4,
        )
        self._loaded = True
        mem = torch.cuda.memory_allocated() / 1024 ** 2
        print(f"[STT]  ✅ Listo en {time.time() - t0:.2f}s  (VRAM: {mem:.0f} MB)")

    def unload_gpu(self) -> None:
        """
        Descarga Whisper de GPU.
        Llama ANTES de enviar la petición a Ollama.
        En Tegra (memoria unificada) la liberación física tarda ~300-500ms.
        """
        if not self._loaded:
            return

        del self.whisper
        self.whisper = None
        self._loaded = False

        gc.collect()                  # 1. GC de Python libera referencias
        torch.cuda.empty_cache()      # 2. Devuelve caché PyTorch al OS
        torch.cuda.synchronize()      # 3. Espera fin de todos los kernels CUDA
        time.sleep(0.5)               # 4. Tegra consolida bloques contiguos

        mem = torch.cuda.memory_allocated() / 1024 ** 2
        estado = "✅" if mem < 50 else "⚠ ALTA"
        print(f"[STT]  🔄 Whisper descargado. VRAM residual: {mem:.0f} MB {estado}")

        if mem > 50:                  # Segunda pasada si queda demasiado
            time.sleep(0.5)
            torch.cuda.empty_cache()

    def reload_gpu(self) -> None:
        """
        Recarga Whisper en GPU.
        Llama DESPUÉS de que TTS termine de reproducir.
        Con KEEP_ALIVE=0 en Docker, Qwen ya se descargó automáticamente.
        """
        if self._loaded:
            return
        time.sleep(0.3)   # Margen para que KEEP_ALIVE=0 termine
        self._load()

    # ── DSP ────────────────────────────────────────────────────────────────

    @staticmethod
    def prepare_audio(
        raw: np.ndarray,
        gain: float = GAIN,
        orig_sr: int = SR_HW,
        target_sr: int = SR_TARGET,
    ) -> np.ndarray:
        """
        Pipeline DSP: ganancia → resampleo 44100→16000 Hz
                    → pre-énfasis → normalización de pico.
        """
        audio = np.clip(raw * gain, -1.0, 1.0)
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
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
            already_16k: True para saltar el DSP (audio ya a 16 kHz).

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
            return "", 0.0   # Silencio — no gastar GPU

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