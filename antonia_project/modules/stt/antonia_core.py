"""
modules/stt/antonia_core.py
Proyecto Antonia — EAFIT
Pipeline: Wake Word (Hey Jarvis) + Silero VAD + Whisper STT (small)
Hardware: Jetson Orin Nano 8GB | Micrófono USB (device 0) a 44100 Hz

Fixes v3:
  - Descarta audio del wake word para que Whisper no lo transcriba
  - VAD más paciente para frases largas (silence_count extendido)
  - Pre-énfasis de señal para recuperar fricativas y consonantes
  - Supresión de doble trigger durante LISTENING
  - Noise gate antes del VAD para ignorar ruido de fondo constante

R-1: Buffer VAD pre-asignado — elimina np.concatenate() en el hot path.
     Evita ~150 heap allocations por utterance en el Cortex-A78AE.
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import torch
import torchaudio
from faster_whisper import WhisperModel
from openwakeword.model import Model
from silero_vad import load_silero_vad

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

MODEL_SIZE = "small"
DEVICE     = "cuda"
COMPUTE    = "int8"

MIC_DEVICE = 0
SR_HW      = 44100
SR_TARGET  = 16000

CHUNK_SAMPLES = 4410   # 100ms @ 44100 Hz

GAIN       = 3.0
MIN_ENERGY = 0.04

WW_THRESHOLD   = 0.35
WW_FLUSH_CHUNKS = 6    # 600ms post-wake-word descartados

VAD_WINDOW        = 512   # Silero exige exactamente 512 samples @ 16kHz
VAD_THRESHOLD     = 0.40
SILENCE_WINDOWS   = 20
MIN_SPEECH_WINDOWS = 3
NOISE_GATE_RMS    = 0.008

LISTENING_TIMEOUT_CHUNKS = 60   # 6 segundos

# R-1: Tamaño del buffer pre-asignado para la acumulación VAD.
# chunk_16k por llamada ≈ CHUNK_SAMPLES * SR_TARGET / SR_HW ≈ 1604 samples.
# 2 × VAD_WINDOW + chunk headroom = 512*2 + 1604 = 2628 → 3000 con margen.
_VAD_BUF_SIZE = 3000

# ══════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════

print("[INIT] Configurando resampler (torchaudio sinc_interp_hann)...")
resampler = torchaudio.transforms.Resample(
    orig_freq=SR_HW,
    new_freq=SR_TARGET,
    resampling_method="sinc_interp_hann",
)

print("[INIT] Cargando Silero VAD...")
silero_vad = load_silero_vad()
silero_vad.reset_states()

print("[INIT] Cargando OpenWakeWord ('hey_jarvis')...")
oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")

print(f"[INIT] Cargando Whisper '{MODEL_SIZE}' ({DEVICE}, {COMPUTE})...")
t0 = time.time()
whisper = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE,
    num_workers=2,
    cpu_threads=4,
)
print(f"[INIT] ✅ Todo listo en {time.time() - t0:.2f}s\n")


# ══════════════════════════════════════════════════════════════
# PROCESADO DE AUDIO
# ══════════════════════════════════════════════════════════════

def resample_chunk(raw: np.ndarray) -> np.ndarray:
    """44100 Hz float32 → 16000 Hz float32 con ganancia aplicada."""
    t = torch.from_numpy(raw * GAIN).float().unsqueeze(0)
    return resampler(t).squeeze(0).numpy()


def preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """
    Pre-énfasis de señal.
    Amplifica frecuencias altas (2kHz-8kHz) donde viven las fricativas
    (s, f, ch, j) y consonantes oclusivas (p, t, k) que el mic USB
    atenúa más que los medios.
    y[n] = x[n] - coef * x[n-1]
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


class VadBuffer:
    """
    R-1: Buffer pre-asignado para la acumulación de samples entre callbacks VAD.

    Reemplaza el patrón leftovers: list + np.concatenate() que ejecutaba
    ~150 heap allocations por utterance. El hot path ahora es un único
    memcpy (buf[n:n+k] = chunk) sin llamadas al allocator de Python.
    """

    def __init__(self, capacity: int = _VAD_BUF_SIZE):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._len = 0

    def reset(self) -> None:
        self._len = 0

    def push(self, chunk: np.ndarray) -> None:
        n = len(chunk)
        end = self._len + n
        if end > len(self._buf):
            # Buffer demasiado pequeño — ampliar (no debería ocurrir en condiciones normales)
            new_buf = np.zeros(end * 2, dtype=np.float32)
            new_buf[:self._len] = self._buf[:self._len]
            self._buf = new_buf
        self._buf[self._len:end] = chunk
        self._len = end

    def score(self) -> float:
        """
        Procesa todas las ventanas de 512 samples disponibles.
        Retorna la probabilidad VAD máxima encontrada en este bloque.
        Los samples residuales (< 512) se preservan para el siguiente push().
        """
        max_prob = 0.0
        offset   = 0

        while offset + VAD_WINDOW <= self._len:
            window = self._buf[offset:offset + VAD_WINDOW]
            t      = torch.from_numpy(window.copy())
            prob   = silero_vad(t, SR_TARGET).item()
            max_prob = max(max_prob, prob)
            offset += VAD_WINDOW

        # Compactar: mover residuo al inicio del buffer
        remainder = self._len - offset
        if remainder > 0:
            self._buf[:remainder] = self._buf[offset:self._len]
        self._len = remainder

        return max_prob


def transcribe(audio_raw_44k: np.ndarray) -> tuple[str, float]:
    """
    Recibe audio crudo a 44100 Hz, lo prepara completamente y transcribe.
    Pipeline interno: resample → pre-énfasis → normalización → Whisper
    """
    audio_16k = resample_chunk(audio_raw_44k)
    audio_16k = preemphasis(audio_16k, coef=0.97)

    peak = np.max(np.abs(audio_16k))
    if peak > 0:
        audio_16k = audio_16k / peak * 0.95

    t0 = time.time()
    segments, _ = whisper.transcribe(
        audio_16k,
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
            "Transcripción de consulta en el Laboratorio de Control Digital, "
            "Universidad EAFIT, Medellín, Colombia. El usuario puede preguntar "
            "sobre PLCs, osciloscopios, conexiones eléctricas, horarios y normas."
        ),
    )
    texto = " ".join(s.text.strip() for s in segments)
    return texto, time.time() - t0


# ══════════════════════════════════════════════════════════════
# COLA DE AUDIO
# ══════════════════════════════════════════════════════════════

_audio_queue: queue.Queue = queue.Queue(maxsize=80)

def _audio_callback(indata, frames, time_info, status):
    """Solo encola. Nunca procesa. Nunca bloquea."""
    if status.input_overflow:
        return
    if status:
        print(f"[HW] {status}")
    try:
        _audio_queue.put_nowait(indata.flatten().copy())
    except queue.Full:
        pass


# ══════════════════════════════════════════════════════════════
# LOOP PRINCIPAL — MÁQUINA DE ESTADOS
# ══════════════════════════════════════════════════════════════

def main_loop():
    """
    Estados:
      SLEEPING   → Escucha solo el wake word.
      FLUSHING   → Descarta N chunks post-wake-word.
      LISTENING  → Graba con Silero VAD. Detecta inicio y fin de utterance.
      PROCESSING → Whisper corriendo en hilo paralelo.
    """
    state         = "SLEEPING"
    raw_buffer    = []
    vad_buf       = VadBuffer()   # R-1: buffer pre-asignado, sin allocations en hot path
    silence_count = 0
    speech_count  = 0
    flush_count   = 0
    total_chunks  = 0

    print("══════════════════════════════════════════════════════")
    print("  Antonia está durmiendo. Di 'Hey Jarvis' para hablar.")
    print("══════════════════════════════════════════════════════\n")

    with sd.InputStream(
        samplerate=SR_HW,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SAMPLES,
        device=MIC_DEVICE,
        latency="high",
        callback=_audio_callback,
    ):
        while True:
            raw_chunk = _audio_queue.get()
            chunk_16k = resample_chunk(raw_chunk)

            # ══════════════════════════════════════════════
            if state == "SLEEPING":
            # ══════════════════════════════════════════════
                chunk_int16 = (np.clip(chunk_16k, -1.0, 1.0) * 32767).astype(np.int16)
                prediction  = oww.predict(chunk_int16)

                if any(score > WW_THRESHOLD for score in prediction.values()):
                    print("\n🔥 Wake word detectada. Preparando escucha...\n")
                    state       = "FLUSHING"
                    flush_count = 0

            # ══════════════════════════════════════════════
            elif state == "FLUSHING":
            # ══════════════════════════════════════════════
                flush_count += 1
                if flush_count >= WW_FLUSH_CHUNKS:
                    print("  🎙️  ¡Habla ahora!\n")
                    state         = "LISTENING"
                    raw_buffer    = []
                    vad_buf.reset()
                    silence_count = 0
                    speech_count  = 0
                    total_chunks  = 0
                    silero_vad.reset_states()

            # ══════════════════════════════════════════════
            elif state == "LISTENING":
            # ══════════════════════════════════════════════
                total_chunks += 1
                raw_buffer.append(raw_chunk)

                rms = np.sqrt(np.mean(chunk_16k ** 2))
                if rms < NOISE_GATE_RMS:
                    if speech_count > 0:
                        silence_count += 1
                else:
                    # R-1: push() es un memcpy en el buffer pre-asignado
                    vad_buf.push(chunk_16k)
                    speech_prob = vad_buf.score()

                    if speech_prob >= VAD_THRESHOLD:
                        speech_count  += 1
                        silence_count  = 0
                    elif speech_count > 0:
                        silence_count += 1

                if speech_count >= MIN_SPEECH_WINDOWS and silence_count >= SILENCE_WINDOWS:
                    captured      = np.concatenate(raw_buffer)
                    raw_buffer    = []
                    vad_buf.reset()
                    silence_count = 0
                    speech_count  = 0

                    print(f"[VAD] Utterance capturado "
                          f"({len(captured)/SR_HW:.1f}s de audio). "
                          f"Transcribiendo...", flush=True)

                    def _run(audio_raw):
                        texto, latencia = transcribe(audio_raw)
                        print("─" * 52)
                        print(f"  [ANTONIA ENTENDIÓ] : {texto or '(silencio)'}")
                        print(f"  [LATENCIA STT]     : {latencia:.3f}s")
                        print("─" * 52)
                        print("\n  Di 'Hey Jarvis' para continuar.\n")

                    threading.Thread(target=_run, args=(captured,), daemon=True).start()
                    state = "SLEEPING"

                elif total_chunks >= LISTENING_TIMEOUT_CHUNKS:
                    print("[VAD] Sin voz detectada. Volviendo a dormir...\n")
                    state         = "SLEEPING"
                    raw_buffer    = []
                    vad_buf.reset()
                    silence_count = 0
                    speech_count  = 0
                    silero_vad.reset_states()


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[INFO] Apagando Antonia...")
