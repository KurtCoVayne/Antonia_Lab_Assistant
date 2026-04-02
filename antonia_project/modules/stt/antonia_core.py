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

# 100ms por chunk — balance entre latencia de respuesta y estabilidad USB
CHUNK_SAMPLES = 4410

GAIN      = 3.0
MIN_ENERGY = 0.04

# — Wake Word —
WW_THRESHOLD = 0.35

# FIX 1: Cuántos chunks descartar tras wake word (elimina el audio de "Hey Jarvis"
# para que Whisper no lo transcriba). 100ms × 6 = 600ms de cola de silencio.
WW_FLUSH_CHUNKS = 6

# — VAD (Silero) —
VAD_WINDOW    = 512    # Silero exige exactamente 512 samples @ 16kHz

# FIX 2: Umbrales más tolerantes para frases largas en entorno con ruido
VAD_THRESHOLD     = 0.40   # Bajado de 0.45 — detecta voz más fácilmente
SILENCE_WINDOWS   = 20     # Subido de 12 → 20 × 32ms = ~640ms antes de cortar
                            # Permite pausas naturales dentro de una frase
MIN_SPEECH_WINDOWS = 3     # Mínimo de voz válida para transcribir (bajado de 4)

# FIX 3: Noise gate — descarta chunks cuya energía sea indistinguible del
# ruido de fondo constante del laboratorio (ventiladores, equipos).
# Si el RMS del chunk está por debajo de este umbral, se trata como silencio
# incluso si Silero lo clasifica como voz.
NOISE_GATE_RMS = 0.008     # Ajustar si el lab tiene mucho ruido de fondo

# Timeout: si no hay voz en N chunks tras wake word → volver a dormir
LISTENING_TIMEOUT_CHUNKS = 60   # 60 × 100ms = 6 segundos

# ══════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════

print("[INIT] Configurando resampler (torchaudio Kaiser)...")
resampler = torchaudio.transforms.Resample(
    orig_freq=SR_HW,
    new_freq=SR_TARGET,
    resampling_method="sinc_interp_kaiser",
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
    FIX 4: Pre-énfasis de señal.
    Amplifica frecuencias altas (2kHz-8kHz) donde viven las fricativas
    (s, f, ch, j) y consonantes oclusivas (p, t, k) que el mic USB
    atenúa más que los medios. Mejora la inteligibilidad de Whisper
    en frases con palabras como 'están', 'frecuencia', 'configure'.
    y[n] = x[n] - coef * x[n-1]
    """
    return np.append(audio[0], audio[1:] - coef * audio[:-1])


def vad_score(chunk_16k: np.ndarray, leftovers: list) -> tuple[float, list]:
    """
    Pasa ventanas de exactamente 512 samples por Silero VAD.
    Acumula residuos entre callbacks para no perder samples.
    Devuelve la probabilidad máxima encontrada en este chunk.
    """
    leftovers.append(chunk_16k)
    combined = np.concatenate(leftovers)
    max_prob = 0.0

    while len(combined) >= VAD_WINDOW:
        window   = combined[:VAD_WINDOW]
        combined = combined[VAD_WINDOW:]
        t        = torch.from_numpy(window.astype(np.float32))
        prob     = silero_vad(t, SR_TARGET).item()
        max_prob = max(max_prob, prob)

    leftovers.clear()
    if len(combined) > 0:
        leftovers.append(combined)

    return max_prob, leftovers


def transcribe(audio_raw_44k: np.ndarray) -> tuple[str, float]:
    """
    Recibe audio crudo a 44100 Hz, lo prepara completamente y transcribe.
    Pipeline interno: resample → pre-énfasis → normalización → Whisper
    """
    # 1. Resamplear todo el buffer capturado
    audio_16k = resample_chunk(audio_raw_44k)

    # 2. Pre-énfasis para recuperar consonantes atenuadas por el mic
    audio_16k = preemphasis(audio_16k, coef=0.97)

    # 3. Normalización de pico
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
            speech_pad_ms=250,       # Subido: más padding al final de la frase
            threshold=0.15,
        ),
        temperature=0.0,
        condition_on_previous_text=False,
        without_timestamps=True,
        # FIX 5: initial_prompt ayuda a Whisper a contextualizar el dominio
        # y a escribir correctamente términos técnicos y nombres propios
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
        return   # Descarte silencioso — mejor perder 100ms que bloquear
    if status:
        print(f"[HW] {status}")
    try:
        _audio_queue.put_nowait(indata.flatten().copy())
    except queue.Full:
        pass     # Cola llena (procesamiento lento) — descarte controlado


# ══════════════════════════════════════════════════════════════
# LOOP PRINCIPAL — MÁQUINA DE ESTADOS
# ══════════════════════════════════════════════════════════════

def main_loop():
    """
    Estados:
      SLEEPING   → Escucha solo el wake word. Liviano, sin grabar.
      FLUSHING   → Descarta N chunks post-wake-word (elimina audio de "Hey Jarvis")
      LISTENING  → Graba con Silero VAD. Detecta inicio y fin de utterance.
      PROCESSING → Whisper corriendo en hilo paralelo. Ya volvimos a SLEEPING.
    """
    state         = "SLEEPING"
    raw_buffer    = []
    vad_leftovers = []
    silence_count = 0
    speech_count  = 0
    flush_count   = 0
    total_chunks  = 0    # Contador para timeout de LISTENING

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
                # FIX 1: Descartar audio del wake word antes de grabar
                # para que Whisper no transcriba "Hey Jarvis" como parte
                # de la pregunta del usuario.
                flush_count += 1
                if flush_count >= WW_FLUSH_CHUNKS:
                    print("  🎙️  ¡Habla ahora!\n")
                    state         = "LISTENING"
                    raw_buffer    = []
                    vad_leftovers = []
                    silence_count = 0
                    speech_count  = 0
                    total_chunks  = 0
                    silero_vad.reset_states()

            # ══════════════════════════════════════════════
            elif state == "LISTENING":
            # ══════════════════════════════════════════════
                total_chunks += 1
                raw_buffer.append(raw_chunk)

                # Noise gate: RMS del chunk a 16kHz
                rms = np.sqrt(np.mean(chunk_16k ** 2))
                if rms < NOISE_GATE_RMS:
                    # Energía de ruido de fondo — tratar como silencio directo
                    if speech_count > 0:
                        silence_count += 1
                else:
                    # Energía suficiente — consultar Silero VAD
                    speech_prob, vad_leftovers = vad_score(chunk_16k, vad_leftovers)

                    if speech_prob >= VAD_THRESHOLD:
                        speech_count  += 1
                        silence_count  = 0
                    elif speech_count > 0:
                        silence_count += 1

                # ── ¿Fin de utterance? ────────────────────────────
                if speech_count >= MIN_SPEECH_WINDOWS and silence_count >= SILENCE_WINDOWS:
                    captured      = np.concatenate(raw_buffer)
                    raw_buffer    = []
                    vad_leftovers = []
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
                    state = "SLEEPING"   # Mic sigue vivo mientras Whisper procesa

                # ── Timeout: no habló en 6 segundos ──────────────
                elif total_chunks >= LISTENING_TIMEOUT_CHUNKS:
                    print("[VAD] Sin voz detectada. Volviendo a dormir...\n")
                    state         = "SLEEPING"
                    raw_buffer    = []
                    vad_leftovers = []
                    silence_count = 0
                    speech_count  = 0
                    silero_vad.reset_states()


if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n[INFO] Apagando Antonia...")