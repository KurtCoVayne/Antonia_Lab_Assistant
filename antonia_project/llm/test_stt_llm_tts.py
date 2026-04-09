"""
tests/test_stt_llm_tts.py
Proyecto Antonia — EAFIT

Pipeline completo de prueba: STT → LLM → TTS
Whisper small GPU float16 | Qwen 2.5 1.5B Docker | Kokoro-82M CPU

Ciclo de memoria GPU (sistema de relevos):
  [IDLE]
    ↓ wake implícito (grabación manual en esta prueba)
  [STT]   Whisper en GPU  (~900 MB VRAM)
    ↓ unload_gpu() + drop_os_cache()
  [LLM]   Qwen en GPU     (~1600 MB VRAM) — heap CUDA limpio
    ↓ KEEP_ALIVE=0 → Qwen se descarga automáticamente
  [TTS]   Kokoro en CPU   (0 MB VRAM) — sin conflicto
    ↓ reload_gpu()
  [IDLE]

Los WAVs de cada turno se guardan en tests/
para verificación auditiva vía SSH.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import subprocess
import asyncio
import time
import httpx
import sounddevice as sd
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Ruta dinámica al proyecto ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.stt.stt_module import AntoniaSTT
from modules.tts.tts_module import tts as antonia_tts

# ── Configuración ──────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/chat"
MODEL          = "qwen2.5:1.5b"
MIC_DEVICE     = 0
SR_HW          = 44100
RECORD_SECONDS = 7
MAX_RETRIES    = 2
N_TURNOS       = 3        # Número de ciclos a probar en el loop

SYSTEM_PROMPT = (
    "Eres Antonia, asistente del Laboratorio de Control Digital de EAFIT. "
    "Responde en español, máximo 3 oraciones cortas, de forma directa. "
    "Usa solo texto plano, sin listas, asteriscos ni emojis. "
    "Si no tienes información, di: No tengo esa información, consulta al monitor."
)

# Directorio donde se guardan los WAVs de cada turno
WAV_DIR = PROJECT_ROOT / "tests" / "pipeline_runs"
WAV_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

def drop_os_cache() -> None:
    """
    Pide al kernel liberar la caché de páginas.
    Ayuda al allocator Tegra a consolidar bloques físicos contiguos
    antes de que Ollama intente reservar memoria para Qwen.
    """
    try:
        subprocess.run(
            ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
            check=True, capture_output=True, timeout=3
        )
        print("[MEM]  Caché del SO liberada ✅")
    except Exception:
        print("[MEM]  No se pudo liberar caché del SO (continúa sin esto)")


def record_audio(seconds: int, sr: int, device: int) -> np.ndarray:
    """Graba audio mono float32 desde el micrófono USB."""
    print(f"\n  🎤 Grabando {seconds}s... ¡Habla ahora!")
    audio = sd.rec(
        int(seconds * sr),
        samplerate=sr,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return audio.flatten()


def wav_path(turno: int, fase: str) -> str:
    """Genera nombre de archivo WAV con timestamp para identificación."""
    ts = datetime.now().strftime("%H%M%S")
    return str(WAV_DIR / f"turno{turno:02d}_{fase}_{ts}.wav")


# ══════════════════════════════════════════════════════════════
# CLIENTE LLM
# ══════════════════════════════════════════════════════════════

async def ask_llm(
    texto: str,
    client: httpx.AsyncClient,
    historial: list[dict],
    retry: int = 0,
) -> tuple[str, float, float]:
    """
    Envía la pregunta a Qwen 2.5 1.5B via Docker Ollama.
    Incluye historial de conversación (máximo 2 turnos anteriores).

    Returns:
        (respuesta, latencia_s, tok_s)
    """
    # Construir mensajes con historial (máximo 4 mensajes = 2 turnos)
    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes += historial[-4:]
    mensajes.append({"role": "user", "content": texto})

    payload = {
        "model":  MODEL,
        "keep_alive": 0,
        "stream": False,
        "options": {
            "num_ctx":        1024,
            "temperature":    0.1,
            "top_p":          0.8,
            "repeat_penalty": 1.15,
            "num_predict":    120,
        },
        "messages": mensajes,
    }

    t0 = time.time()
    try:
        resp    = await client.post(OLLAMA_URL, json=payload)
        data    = resp.json()
        elapsed = time.time() - t0

        if "error" in data:
            err = data["error"]
            print(f"\n  ❌ OLLAMA ERROR: {err}")
            if "out of memory" in err and retry < MAX_RETRIES:
                print(f"  ⏳ OOM residual — reintentando en 2s ({retry+1}/{MAX_RETRIES})")
                await asyncio.sleep(2.0)
                drop_os_cache()
                return await ask_llm(texto, client, historial, retry + 1)
            return "", elapsed, 0.0

        eval_c    = data.get("eval_count", 0)
        eval_d    = data.get("eval_duration", 1) / 1e9
        tok_s     = eval_c / eval_d if eval_d > 0 else 0.0
        respuesta = data.get("message", {}).get("content", "").strip()
        return respuesta, elapsed, tok_s

    except httpx.TimeoutException:
        print(f"  ❌ TIMEOUT ({time.time()-t0:.1f}s)")
        return "", time.time() - t0, 0.0

    except Exception as e:
        print(f"  ❌ Error inesperado: {type(e).__name__}: {e}")
        return "", time.time() - t0, 0.0


# ══════════════════════════════════════════════════════════════
# UN CICLO COMPLETO STT → LLM → TTS
# ══════════════════════════════════════════════════════════════

async def run_turn(
    turno: int,
    stt: AntoniaSTT,
    client: httpx.AsyncClient,
    historial: list[dict],
) -> dict:
    """
    Ejecuta un ciclo completo del pipeline con sistema de relevos GPU.

    Orden de memoria GPU:
      1. STT:  Whisper en GPU
      2. unload_gpu() + drop_os_cache()
      3. LLM:  Qwen en GPU (heap limpio)
      4. KEEP_ALIVE=0 → Qwen se descarga automáticamente al terminar
      5. TTS:  Kokoro en CPU (sin conflicto GPU)
      6. reload_gpu() → Whisper vuelve para el siguiente turno

    Returns:
        dict con texto, respuesta, tts_path y métricas de latencia
    """
    t_ciclo = time.time()
    resultado = {
        "turno":     turno,
        "texto":     "",
        "respuesta": "",
        "tts_path":  None,
        "lat_stt":   0.0,
        "lat_llm":   0.0,
        "lat_tts":   0.0,
        "tok_s":     0.0,
        "lat_total": 0.0,
        "ok":        False,
    }

    # ── FASE 1: Captura de audio ────────────────────────────────────────
    raw = record_audio(RECORD_SECONDS, SR_HW, MIC_DEVICE)

    # ── FASE 2: STT — Whisper en GPU ────────────────────────────────────
    print(f"\n  [STT] Transcribiendo...")
    texto, lat_stt = stt.transcribe(raw)
    resultado["texto"]   = texto
    resultado["lat_stt"] = lat_stt
    print(f"  [STT] ({lat_stt:.2f}s): \"{texto}\"")

    if len(texto.strip()) < 2:
        print("  ⚠  Silencio o frase muy corta — saltando LLM y TTS.")
        resultado["lat_total"] = time.time() - t_ciclo
        return resultado

    # ── FASE 3: RELEVO — Whisper sale de GPU ────────────────────────────
    print("\n  [RELEVO] Descargando Whisper de GPU...")
    stt.unload_gpu()
    drop_os_cache()

    # ── FASE 4: LLM — Qwen entra en GPU ─────────────────────────────────
    print(f"  [LLM] Consultando {MODEL}...")
    respuesta, lat_llm, tok_s = await ask_llm(texto, client, historial)
    resultado["respuesta"] = respuesta
    resultado["lat_llm"]   = lat_llm
    resultado["tok_s"]     = tok_s

    if respuesta:
        print(f"  [LLM] ({lat_llm:.2f}s, {tok_s:.1f} tok/s):")
        print(f"        \"{respuesta}\"")
        # Acumular historial para siguiente turno
        historial.append({"role": "user",      "content": texto})
        historial.append({"role": "assistant", "content": respuesta})
    else:
        print(f"  ⚠  LLM sin respuesta ({lat_llm:.2f}s)")

    # ── FASE 5: TTS — Kokoro en CPU (Qwen ya se descargó por KEEP_ALIVE=0)
    if respuesta and antonia_tts is not None:
        print("\n  [TTS] Sintetizando respuesta...")
        t_tts = time.time()
        wav   = wav_path(turno, "respuesta")

        # save_wav=True siempre en pruebas — para verificar por SSH
        antonia_tts.speak(
            respuesta,
            save_wav=True,
            wav_filename=f"turno{turno:02d}_respuesta.wav",
            play_audio=True,
        )
        resultado["lat_tts"]  = time.time() - t_tts
        resultado["tts_path"] = str(WAV_DIR / f"turno{turno:02d}_respuesta.wav")
    elif antonia_tts is None:
        print("  ⚠  TTS no inicializado — respuesta solo en texto.")

    # ── FASE 6: RELEVO — Whisper vuelve a GPU ───────────────────────────
    print("\n  [RELEVO] Recargando Whisper en GPU...")
    stt.reload_gpu()

    resultado["lat_total"] = time.time() - t_ciclo
    resultado["ok"]        = bool(respuesta)
    return resultado


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

async def main():
    ts_inicio = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("═" * 60)
    print("  PIPELINE ANTONIA — STT + LLM + TTS")
    print(f"  {ts_inicio}")
    print("  Whisper small GPU float16")
    print("  Qwen 2.5 1.5B Docker (dustynv/ollama)")
    print("  Kokoro-82M CPU")
    print("═" * 60)

    # ── Inicializar módulos ───────────────────────────────────────────────
    print("\n[INIT] Cargando STT...")
    stt = AntoniaSTT()

    if antonia_tts is None:
        print("[WARN] TTS no disponible — el pipeline continuará sin voz.")

    # ── Verificar Ollama Docker ───────────────────────────────────────────
    historial: list[dict] = []
    metricas:  list[dict] = []

    async with httpx.AsyncClient(timeout=60.0) as client:

        print("\n[INIT] Verificando Ollama Docker...")
        try:
            r       = await client.get("http://localhost:11434/api/tags")
            modelos = [m["name"] for m in r.json().get("models", [])]
            print(f"[OK]   Ollama activo. Modelos disponibles: {modelos}")
            if not any(MODEL.split(":")[0] in m for m in modelos):
                print(f"[WARN] Modelo '{MODEL}' no encontrado.")
                print(f"       Ejecuta: sudo docker exec -it ollama_antonia ollama pull {MODEL}")
        except Exception as e:
            print(f"[ERROR] Ollama no responde en localhost:11434: {e}")
            print("        Ejecuta: sudo docker start ollama_antonia")
            return

        # ── Loop de turnos ────────────────────────────────────────────────
        for turno in range(1, N_TURNOS + 1):
            print(f"\n{'─'*60}")
            print(f"  TURNO {turno}/{N_TURNOS}")
            print(f"{'─'*60}")

            m = await run_turn(turno, stt, client, historial)
            metricas.append(m)

            print(f"\n  ⏱️  MÉTRICAS TURNO {turno}:")
            print(f"     STT:   {m['lat_stt']:.2f}s")
            print(f"     LLM:   {m['lat_llm']:.2f}s  ({m['tok_s']:.1f} tok/s)")
            print(f"     TTS:   {m['lat_tts']:.2f}s")
            print(f"     TOTAL: {m['lat_total']:.2f}s")
            if m["tts_path"]:
                print(f"     WAV:   {m['tts_path']}")

            if turno < N_TURNOS:
                print(f"\n  Siguiente turno en 3s...")
                await asyncio.sleep(3.0)

    # ── Resumen final ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("  RESUMEN FINAL")
    print(f"{'═'*60}")

    ok = [m for m in metricas if m["ok"]]

    if ok:
        def avg(key):
            return sum(m[key] for m in ok) / len(ok)

        print(f"  Turnos exitosos:         {len(ok)}/{N_TURNOS}")
        print(f"  Latencia STT promedio:   {avg('lat_stt'):.2f}s")
        print(f"  Latencia LLM promedio:   {avg('lat_llm'):.2f}s  "
              f"({avg('tok_s'):.1f} tok/s avg)")
        print(f"  Latencia TTS promedio:   {avg('lat_tts'):.2f}s")
        print(f"  Latencia TOTAL promedio: {avg('lat_total'):.2f}s")
        print(f"  {'✅ Pipeline estable' if len(ok) == N_TURNOS else '⚠ Algunos turnos fallaron'}")

        wavs = [m["tts_path"] for m in ok if m["tts_path"]]
        if wavs:
            print(f"\n  WAVs generados ({len(wavs)}):")
            for w in wavs:
                print(f"    {w}")
            print(f"\n  Descargar vía SSH:")
            print(f"    scp mecatronica@<IP>:{WAV_DIR}/*.wav .")
    else:
        print("  ❌ Ningún turno completó el pipeline completo.")
        print("     Revisa los logs y verifica que Ollama Docker está activo.")

    print(f"{'═'*60}\n")


if __name__ == "__main__":
    asyncio.run(main())