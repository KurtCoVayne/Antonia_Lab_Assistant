"""
tests/test_stt_llm_tts.py
Proyecto Antonia — EAFIT

Pipeline completo de prueba: STT → LLM → TTS
Whisper small GPU | Qwen 2.5 1.5B Docker | Kokoro-82M CPU/GPU

Ciclo de memoria GPU (sistema de relevos):
  [IDLE]
  [STT]   Whisper en GPU  (~900 MB VRAM)
    ↓ unload_gpu() + drop_os_cache() en paralelo (asyncio.gather)
  [LLM]   Qwen en GPU     (~1600 MB VRAM)
    ↓ KEEP_ALIVE expira → Qwen se descarga automáticamente
  [TTS]   Kokoro en CPU/GPU (sin conflicto)
    ↓ reload_gpu()
  [IDLE]

Los WAVs de cada turno se guardan en tests/pipeline_runs/
para verificación auditiva vía SSH.
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "max_split_size_mb:512,"
    "garbage_collection_threshold:0.8"
)

import sys
import random
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
N_TURNOS       = 3

# C-2: keep_alive=0 evicta Qwen tras cada respuesta (relay estricto).
# Aumentar a ~10 si se acepta que Qwen permanezca en VRAM durante TTS.
# Con 0, el cold-load de Qwen desde el overlay filesystem del contenedor
# domina la latencia por turno (8-12s en NVMe frío).
OLLAMA_KEEP_ALIVE_S = 0

SYSTEM_PROMPT = (
    "Eres Antonia, asistente del Laboratorio de Control Digital de EAFIT. "
    "Responde en español, máximo 3 oraciones cortas, de forma directa. "
    "Usa solo texto plano, sin listas, asteriscos ni emojis. "
    "Si no tienes información, di: No tengo esa información, consulta al monitor."
)

WAV_DIR = PROJECT_ROOT / "tests" / "pipeline_runs"
WAV_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════

async def drop_os_cache() -> None:
    """
    C-3: Pide al kernel liberar la caché de páginas.
    - Completamente asíncrono (asyncio.to_thread + asyncio.wait_for).
    - Timeout duro de 5s sobre el hilo; timeout de 4s sobre el subprocess.
    - Falla silenciosa en todos los casos: el pipeline continúa.
    - Requiere NOPASSWD en sudoers para el comando específico:
        mecatronica ALL=(ALL) NOPASSWD: /bin/sh -c sync && echo 3 > /proc/sys/vm/drop_caches
    """
    def _sync_call() -> None:
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "sync && echo 3 > /proc/sys/vm/drop_caches"],
                check=True,
                capture_output=True,
                timeout=4,
            )
            print("[MEM]  Caché del SO liberada ✅")
        except subprocess.TimeoutExpired:
            print("[MEM]  drop_os_cache: subprocess timeout (4s) — continúa")
        except subprocess.CalledProcessError as e:
            print(f"[MEM]  drop_os_cache: error de proceso ({e.returncode}) — continúa")
        except PermissionError:
            print("[MEM]  drop_os_cache: permiso denegado — configura sudoers NOPASSWD")
        except OSError as e:
            print(f"[MEM]  drop_os_cache: OSError ({e}) — continúa")

    try:
        await asyncio.wait_for(asyncio.to_thread(_sync_call), timeout=5.0)
    except asyncio.TimeoutError:
        print("[MEM]  drop_os_cache: timeout del event loop (5s) — continúa")


def _record_audio_sync(seconds: int, sr: int, device: int) -> np.ndarray:
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

    R-4: Backoff exponencial con jitter en reintentos por OOM.
         Verifica VRAM antes de reintentar.

    Returns:
        (respuesta, latencia_s, tok_s)
    """
    import torch

    mensajes = [{"role": "system", "content": SYSTEM_PROMPT}]
    mensajes += historial[-4:]
    mensajes.append({"role": "user", "content": texto})

    payload = {
        "model":      MODEL,
        "keep_alive": OLLAMA_KEEP_ALIVE_S,
        "stream":     False,
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
                # R-4: backoff exponencial con jitter (1-2s, 3-5s, …)
                delay = (2 ** retry) + random.uniform(0.0, 1.0)
                print(f"  ⏳ OOM — reintentando en {delay:.1f}s ({retry+1}/{MAX_RETRIES})")
                await asyncio.sleep(delay)

                # R-4: verificar VRAM antes de reintentar — no tiene sentido
                # hacer la petición HTTP si el allocator de Tegra no consolidó
                mem_mb = torch.cuda.memory_reserved() / 1024 ** 2
                print(f"[MEM]  VRAM reservada pre-retry: {mem_mb:.0f} MB")
                await drop_os_cache()
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

    C-1: Todas las llamadas bloqueantes se delegan a asyncio.to_thread()
    para no congelar el event loop de asyncio durante operaciones GPU,
    I/O de audio, o subprocesos. unload_gpu() y drop_os_cache() se
    ejecutan en paralelo via asyncio.gather() porque son independientes
    (allocator PyTorch vs. caché de páginas del kernel).

    Orden de memoria GPU:
      1. STT:  Whisper en GPU
      2. unload_gpu() ‖ drop_os_cache()  ← en paralelo
      3. LLM:  Qwen en GPU
      4. KEEP_ALIVE expira → Qwen descargado automáticamente
      5. TTS:  Kokoro en CPU/GPU
      6. reload_gpu()
    """
    t_ciclo   = time.time()
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

    # ── FASE 1: Captura de audio (bloqueante → hilo) ────────────────────
    raw = await asyncio.to_thread(_record_audio_sync, RECORD_SECONDS, SR_HW, MIC_DEVICE)

    # ── FASE 2: STT — Whisper en GPU (bloqueante → hilo) ────────────────
    print(f"\n  [STT] Transcribiendo...")
    texto, lat_stt = await asyncio.to_thread(stt.transcribe, raw)
    resultado["texto"]   = texto
    resultado["lat_stt"] = lat_stt
    print(f"  [STT] ({lat_stt:.2f}s): \"{texto}\"")

    if len(texto.strip()) < 2:
        print("  ⚠  Silencio o frase muy corta — saltando LLM y TTS.")
        resultado["lat_total"] = time.time() - t_ciclo
        return resultado

    # ── FASE 3: RELEVO — Whisper sale, OS cache se libera (en paralelo) ─
    print("\n  [RELEVO] Descargando Whisper de GPU...")
    await asyncio.gather(
        asyncio.to_thread(stt.unload_gpu),
        drop_os_cache(),
    )

    # ── FASE 4: LLM — Qwen entra en GPU ─────────────────────────────────
    print(f"  [LLM] Consultando {MODEL}...")
    respuesta, lat_llm, tok_s = await ask_llm(texto, client, historial)
    resultado["respuesta"] = respuesta
    resultado["lat_llm"]   = lat_llm
    resultado["tok_s"]     = tok_s

    if respuesta:
        print(f"  [LLM] ({lat_llm:.2f}s, {tok_s:.1f} tok/s):")
        print(f"        \"{respuesta}\"")
        historial.append({"role": "user",      "content": texto})
        historial.append({"role": "assistant", "content": respuesta})
    else:
        print(f"  ⚠  LLM sin respuesta ({lat_llm:.2f}s)")

    # ── FASE 5: TTS — Kokoro en CPU/GPU (Qwen se descargó por KEEP_ALIVE)
    if respuesta and antonia_tts is not None:
        print("\n  [TTS] Sintetizando respuesta...")
        t_tts    = time.time()
        wav_name = f"turno{turno:02d}_respuesta.wav"

        await asyncio.to_thread(
            antonia_tts.speak,
            respuesta,
            True,       # save_wav
            wav_name,
            True,       # play_audio
        )
        resultado["lat_tts"]  = time.time() - t_tts
        resultado["tts_path"] = str(WAV_DIR / wav_name)
    elif antonia_tts is None:
        print("  ⚠  TTS no inicializado — respuesta solo en texto.")

    # ── FASE 6: RELEVO — Whisper vuelve a GPU (bloqueante → hilo) ───────
    print("\n  [RELEVO] Recargando Whisper en GPU...")
    await asyncio.to_thread(stt.reload_gpu)

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
    print("  Whisper small GPU")
    print("  Qwen 2.5 1.5B Docker (dustynv/ollama)")
    print(f"  keep_alive={OLLAMA_KEEP_ALIVE_S}s")
    print("  Kokoro-82M")
    print("═" * 60)

    print("\n[INIT] Cargando STT...")
    stt = await asyncio.to_thread(AntoniaSTT)

    if antonia_tts is None:
        print("[WARN] TTS no disponible — el pipeline continuará sin voz.")

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
