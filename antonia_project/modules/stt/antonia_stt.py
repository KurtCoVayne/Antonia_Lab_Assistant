"""
modules/stt/antonia_stt.py
Proyecto Antonia — EAFIT
Módulo STT: Whisper small GPU float16 con gestión de VRAM para sistema de relevos.

Fix v3:
  - gc.collect() explícito antes de empty_cache()
  - sleep calibrado para que Tegra libere bloques físicos
  - Verificación de liberación real con torch.cuda.memory_allocated()
  - Ganancia + pre-énfasis + normalización integrados en transcribe()
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
MODEL_SIZE = "small"
DEVICE     = "cuda"
COMPUTE    = "float16"   # float16 nativo Tegra — más estable que int8

SR_HW      = 44100       # Sample rate nativo del mic USB
SR_TARGET  = 16000       # Sample rate requerido por Whisper
GAIN       = 3.0         # Compensación de baja sensibilidad del mic


class AntoniaSTT:
    """
    Encapsula Whisper con gestión explícita de ciclo de vida en VRAM.
    Diseñado para el sistema de relevos GPU de Antonia:
      stt.unload_gpu()  → antes de llamar a Ollama
      stt.reload_gpu()  → después de recibir respuesta de Ollama
    """

    def __init__(self):
        self._whisper_loaded = False
        self.whisper = None
        self._load_whisper()

    # ── Carga / descarga ───────────────────────────────────────────────────

    def _load_whisper(self) -> None:
        print(f"[STT]  Cargando Whisper '{MODEL_SIZE}' ({DEVICE}, {COMPUTE})...")
        t0 = time.time()
        self.whisper = WhisperModel(
            MODEL_SIZE,
            device=DEVICE,
            compute_type=COMPUTE,
            device_index=0,    # Ruteo explícito al bus GPU principal
            num_workers=1,     # 1 worker evita clonación de tensores en RAM
            cpu_threads=4,
        )
        self._whisper_loaded = True
        mem_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"[STT]  ✅ Whisper cargado en {time.time()-t0:.2f}s "
              f"(VRAM usada: {mem_mb:.0f} MB)")

    def unload_gpu(self) -> None:
        """
        Descarga Whisper de GPU y espera que CUDA libere bloques físicos.
        En Tegra (memoria unificada) esto puede tardar 300-500ms.
        LLAMAR ANTES de enviar petición a Ollama.
        """
        if not self._whisper_loaded:
            return

        del self.whisper
        self.whisper = None
        self._whisper_loaded = False

        # Orden crítico en Tegra para liberar bloques contiguos:
        gc.collect()                    # 1. Python GC — libera referencias
        torch.cuda.empty_cache()        # 2. Devuelve caché PyTorch al OS
        torch.cuda.synchronize()        # 3. Espera que todos los kernels CUDA terminen

        # 4. Pausa para que el allocator Tegra consolide bloques libres
        # Sin este sleep, Ollama puede llegar antes de que los bloques estén disponibles
        time.sleep(0.5)

        mem_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"[STT]  🔄 Whisper descargado. VRAM residual: {mem_mb:.0f} MB "
              f"{'✅' if mem_mb < 50 else '⚠ ALTA — esperar más'}")

        # Si queda demasiada memoria, esperar otro ciclo
        if mem_mb > 50:
            print("[STT]  Esperando liberación adicional de VRAM...")
            time.sleep(0.5)
            torch.cuda.empty_cache()

    def reload_gpu(self) -> None:
        """
        Recarga Whisper en GPU. LLAMAR DESPUÉS de recibir respuesta de Ollama.
        Con KEEP_ALIVE=0 en Docker, Qwen ya se descargó automáticamente.
        """
        if self._whisper_loaded:
            return
        # Pequeña pausa para que KEEP_ALIVE=0 termine de liberar VRAM de Qwen
        time.sleep(0.3)
        self._load_whisper()

    # ── Procesado de audio ─────────────────────────────────────────────────

    @staticmethod
    def prepare_audio(
        raw_44k: np.ndarray,
        gain: float = GAIN,
        orig_sr: int = SR_HW,
        target_sr: int = SR_TARGET,
    ) -> np.ndarray:
        """
        Pipeline DSP completo:
          1. Ganancia (compensación de baja sensibilidad del mic)
          2. Resampleo 44100 → 16000 Hz con librosa (alta calidad)
          3. Pre-énfasis (amplifica fricativas y consonantes)
          4. Normalización de pico
        """
        # 1. Ganancia
        audio = np.clip(raw_44k * gain, -1.0, 1.0)

        # 2. Resampleo
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

        # 3. Pre-énfasis y[n] = x[n] - 0.97*x[n-1]
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])

        # 4. Normalización de pico a 0.95
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak * 0.95

        return audio

    def transcribe(self, audio: np.ndarray, already_16k: bool = False) -> tuple[str, float]:
        """
        Transcribe audio. Acepta tanto 44100 Hz (procesa automáticamente)
        como 16000 Hz (pasar already_16k=True para saltar DSP).

        Returns:
            (texto_transcrito, latencia_segundos)
        """
        if not self._whisper_loaded:
            raise RuntimeError(
                "[STT] Whisper no está en GPU. Llamar reload_gpu() primero."
            )

        # DSP si el audio viene crudo del micrófono
        if not already_16k:
            audio = self.prepare_audio(audio)

        # Verificar energía mínima
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
            temperature=0.0,             # Determinista — sin alucinaciones aleatorias
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=(
                "Transcripción en el Laboratorio de Control Digital, "
                "Universidad EAFIT, Medellín. Puede incluir términos como "
                "PLC, osciloscopio, multímetro, conexión, EAFIT."
            ),
        )
        texto = " ".join(s.text.strip() for s in segments)
        return texto, time.time() - t0