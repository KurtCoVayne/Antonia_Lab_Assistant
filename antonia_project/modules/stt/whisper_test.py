"""
modules/stt/whisper_test.py
Proyecto Antonia — EAFIT
Prueba y benchmark de Whisper con micrófono USB a 44100 Hz.
Hardware: Jetson Orin Nano 8GB | Micrófono USB (device 0)
"""

import time
import sounddevice as sd
import numpy as np
import librosa
from faster_whisper import WhisperModel

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════
MODEL_SIZE  = "small"   # Opciones: "tiny" | "base" | "small"
DEVICE      = "cuda"    # GPU de la Jetson
COMPUTE     = "int8"    # Cuantización INT8 — menor RAM, igual precisión
MIC_DEVICE  = 0         # ID del micrófono USB (verificar con: python -m sounddevice)
SR_HW       = 44100     # Sample rate nativo del micrófono USB
SR_WHISPER  = 16000     # Sample rate requerido por Whisper
DURATION    = 7         # Segundos de grabación
GAIN        = 3.0       # Ganancia por baja sensibilidad del mic
MIN_ENERGY  = 0.05      # Umbral mínimo de energía para considerar audio válido

# ══════════════════════════════════════════════════════════════
# CARGA DEL MODELO
# ══════════════════════════════════════════════════════════════
print(f"[INIT] Cargando Whisper '{MODEL_SIZE}' en {DEVICE} ({COMPUTE})...")
t_load = time.time()
model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE,
    num_workers=2,          # Paralelismo en decodificación
    cpu_threads=4,          # Hilos CPU para pre/post-procesado
)
print(f"[INIT] Modelo listo en {time.time() - t_load:.2f}s\n")


# ══════════════════════════════════════════════════════════════
# CAPTURA DE AUDIO
# ══════════════════════════════════════════════════════════════
def capture_audio(duration: float, sr: int, device: int) -> np.ndarray:
    """Graba audio mono float32 desde el dispositivo USB."""
    print(f"[MIC]  Grabando {duration}s a {sr} Hz... ¡Habla ahora!")
    audio = sd.rec(
        int(duration * sr),
        samplerate=sr,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten()


# ══════════════════════════════════════════════════════════════
# PROCESADO DE SEÑAL (DSP)
# ══════════════════════════════════════════════════════════════
def prepare_audio(audio: np.ndarray, gain: float, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Aplica ganancia, remuestrea de SR_HW → SR_WHISPER
    y normaliza el rango a [-1, 1].
    """
    # 1. Ganancia para compensar baja sensibilidad del micrófono
    audio = np.clip(audio * gain, -1.0, 1.0)

    # 2. Remuestreo 44100 Hz → 16000 Hz (requerido por Whisper)
    audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

    # 3. Normalización final (por si el clip dejó artefactos)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.95   # headroom de 5%

    return audio


# ══════════════════════════════════════════════════════════════
# TRANSCRIPCIÓN
# ══════════════════════════════════════════════════════════════
def transcribe(audio: np.ndarray) -> tuple[str, float]:
    """
    Transcribe audio numpy float32 a 16 kHz.
    Devuelve (texto, latencia_segundos).
    """
    t0 = time.time()
    segments, info = model.transcribe(
        audio,
        language="es",
        beam_size=5,                # Más candidatos → más precisión (era 3)
        best_of=5,                  # Muestras en decoding greedy fallback
        vad_filter=True,            # Elimina silencios con VAD interno
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,      # Padding para no cortar palabras
            threshold=0.15,         # Más sensible que 0.5 para lab ruidoso
        ),
        temperature=0.0,            # Determinista — sin aleatoriedad
        condition_on_previous_text=False,  # Cada segmento independiente
        without_timestamps=True,    # Ahorra cómputo si no necesitas timestamps
    )
    texto = " ".join(s.text.strip() for s in segments)
    return texto, time.time() - t0


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # 1. Captura
    raw = capture_audio(DURATION, SR_HW, MIC_DEVICE)

    # 2. DSP
    audio_16k = prepare_audio(raw, GAIN, SR_HW, SR_WHISPER)

    # 3. Validación de energía
    energy = np.max(np.abs(audio_16k))
    print(f"[DSP]  Nivel de energía: {energy:.3f}", end="  ")
    if energy < MIN_ENERGY:
        print("❌  Señal muy baja — acércate al micrófono o sube la ganancia.")
    else:
        print("✅  Nivel óptimo")

        # 4. Transcripción
        texto, latencia = transcribe(audio_16k)

        print("─" * 45)
        print(f"  TEXTO    : {texto if texto else '(silencio detectado)'}")
        print(f"  LATENCIA : {latencia:.3f}s")
        print("─" * 45)