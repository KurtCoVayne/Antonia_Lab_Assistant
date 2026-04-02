"""
tests/test_stt_llm_pipeline.py
Proyecto Antonia — EAFIT
Pipeline de prueba: STT (Whisper GPU) + Sistema de Relevos + LLM (Qwen Docker)

Fix v3:
  - Uso correcto de /api/chat en lugar de /api/generate
  - Drop de caché del SO antes de llamar a Ollama
  - Retry automático si Ollama falla por OOM residual
  - Modo loop: N preguntas consecutivas para probar estabilidad
  - Métricas detalladas por fase
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

# Ajuste de ruta dinámico
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.stt.antonia_stt import AntoniaSTT

# ── Configuración ──────────────────────────────────────────────────────────
OLLAMA_URL     = "http://localhost:11434/api/chat"   # /api/chat — endpoint correcto
MODEL          = "qwen2.5:1.5b"
MIC_DEVICE     = 0
SR_HW          = 44100
RECORD_SECONDS = 7
MAX_RETRIES    = 2    # Reintentos si Ollama falla por OOM residual

SYSTEM_PROMPT = """Eres Antonia, asistente del Laboratorio de Control Digital de EAFIT.
Responde en español, máximo 3 oraciones cortas, de forma directa.
Usa solo texto plano, sin listas, asteriscos ni emojis.
Si no tienes información, di: No tengo esa información, consulta al monitor."""


# ── Utilidades ────────────────────────────────────────────────────────────

def drop_os_cache() -> None:
    """
    Sugiere al kernel liberar la caché de páginas.
    Ayuda a que Ollama encuentre bloques físicos contiguos en Tegra.
    Requiere sudo — si falla, se ignora silenciosamente.
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


# ── Cliente LLM ───────────────────────────────────────────────────────────

async def ask_llm(
    texto: str,
    client: httpx.AsyncClient,
    retry: int = 0,
) -> tuple[str, float, float]:
    """
    Envía pregunta a Qwen via /api/chat (endpoint correcto para modelos de chat).
    Devuelve (respuesta, latencia_s, tok_s).
    Reintenta automáticamente si hay OOM residual.
    """
    payload = {
        "model":  MODEL,
        "stream": False,
        "options": {
            "num_ctx":        1024,
            "temperature":    0.1,    # Bajo para mínimas alucinaciones en 1.5B
            "top_p":          0.8,
            "repeat_penalty": 1.15,
            "num_predict":    120,    # ~3 oraciones cortas
        },
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": texto},
        ],
    }

    t0 = time.time()
    try:
        resp = await client.post(OLLAMA_URL, json=payload)
        data = resp.json()
        elapsed = time.time() - t0

        if "error" in data:
            err = data["error"]
            print(f"\n  ❌ ERROR OLLAMA: {err}")

            # OOM residual: esperar y reintentar
            if "out of memory" in err and retry < MAX_RETRIES:
                print(f"  ⏳ OOM residual — esperando 2s y reintentando "
                      f"({retry+1}/{MAX_RETRIES})...")
                await asyncio.sleep(2.0)
                drop_os_cache()
                return await ask_llm(texto, client, retry + 1)

            return "", elapsed, 0.0

        eval_c = data.get("eval_count", 0)
        eval_d = data.get("eval_duration", 1) / 1e9   # ns → s
        tok_s  = eval_c / eval_d if eval_d > 0 else 0.0

        respuesta = data.get("message", {}).get("content", "").strip()
        return respuesta, elapsed, tok_s

    except httpx.TimeoutException:
        elapsed = time.time() - t0
        print(f"  ❌ TIMEOUT después de {elapsed:.1f}s")
        return "", elapsed, 0.0

    except Exception as e:
        elapsed = time.time() - t0
        print(f"  ❌ Error inesperado: {type(e).__name__}: {e}")
        return "", elapsed, 0.0


# ── Pipeline principal ────────────────────────────────────────────────────

async def run_single_turn(stt: AntoniaSTT, client: httpx.AsyncClient) -> dict:
    """
    Ejecuta un ciclo completo: Grabar → STT → Relevo → LLM → Relevo.
    Devuelve métricas del ciclo.
    """
    t_ciclo = time.time()

    # ── FASE 1: Captura y STT ─────────────────────────────────────────────
    raw = record_audio(RECORD_SECONDS, SR_HW, MIC_DEVICE)
    texto, lat_stt = stt.transcribe(raw)   # prepare_audio() incluido en transcribe()

    print(f"\n  🗣️  STT ({lat_stt:.2f}s): \"{texto}\"")

    if len(texto.strip()) < 2:
        print("  ⚠  Audio muy corto o silencio. Saltando fase LLM.")
        return {"texto": "", "respuesta": "", "lat_stt": lat_stt,
                "lat_llm": 0.0, "tok_s": 0.0, "lat_total": lat_stt}

    # ── FASE 2: Relevo — Whisper sale de GPU ──────────────────────────────
    print("\n  🔄 [RELEVO] Descargando Whisper de GPU...")
    stt.unload_gpu()

    # Liberar caché del SO para que Ollama encuentre bloques contiguos
    drop_os_cache()

    # ── FASE 3: LLM ───────────────────────────────────────────────────────
    print(f"  🧠 Consultando {MODEL}...")
    respuesta, lat_llm, tok_s = await ask_llm(texto, client)

    if respuesta:
        print(f"  🤖 LLM ({lat_llm:.2f}s, {tok_s:.1f} tok/s):")
        print(f"     \"{respuesta}\"")
    else:
        print(f"  ⚠  LLM no respondió ({lat_llm:.2f}s)")

    # ── FASE 4: Relevo — Whisper vuelve a GPU ─────────────────────────────
    print("\n  🔄 [RELEVO] Recargando Whisper en GPU...")
    stt.reload_gpu()

    lat_total = time.time() - t_ciclo
    return {
        "texto":     texto,
        "respuesta": respuesta,
        "lat_stt":   lat_stt,
        "lat_llm":   lat_llm,
        "tok_s":     tok_s,
        "lat_total": lat_total,
    }


async def main():
    print("═" * 58)
    print("  PIPELINE ANTONIA — TEST STT + LLM")
    print("  Whisper small GPU float16 | Qwen 2.5 1.5B Docker")
    print("═" * 58)

    # Inicializar STT
    stt = AntoniaSTT()

    # Configurar cuántos turnos probar
    N_TURNOS = 3
    metricas = []

    async with httpx.AsyncClient(timeout=60.0) as client:

        # Verificar que Ollama Docker responde antes de empezar
        try:
            r = await client.get("http://localhost:11434/api/tags")
            modelos = [m["name"] for m in r.json().get("models", [])]
            print(f"\n[OK]   Ollama activo. Modelos: {modelos}")
            if MODEL not in modelos and not any("qwen" in m for m in modelos):
                print(f"[WARN] Modelo '{MODEL}' no encontrado. Disponibles: {modelos}")
        except Exception as e:
            print(f"[ERROR] Ollama no responde: {e}")
            print("        Ejecuta: sudo docker start ollama_antonia")
            return

        # Loop de N turnos
        for turno in range(1, N_TURNOS + 1):
            print(f"\n{'─'*58}")
            print(f"  TURNO {turno}/{N_TURNOS}")
            print(f"{'─'*58}")

            m = await run_single_turn(stt, client)
            metricas.append(m)

            print(f"\n  ⏱️  MÉTRICAS TURNO {turno}:")
            print(f"     STT:   {m['lat_stt']:.2f}s")
            print(f"     LLM:   {m['lat_llm']:.2f}s  ({m['tok_s']:.1f} tok/s)")
            print(f"     TOTAL: {m['lat_total']:.2f}s")

            if turno < N_TURNOS:
                print(f"\n  Preparando siguiente turno en 2s...")
                await asyncio.sleep(2.0)

    # Resumen final
    print(f"\n{'═'*58}")
    print("  RESUMEN FINAL")
    print(f"{'═'*58}")
    turnos_ok = [m for m in metricas if m["respuesta"]]
    if turnos_ok:
        avg_stt   = sum(m["lat_stt"]   for m in turnos_ok) / len(turnos_ok)
        avg_llm   = sum(m["lat_llm"]   for m in turnos_ok) / len(turnos_ok)
        avg_total = sum(m["lat_total"] for m in turnos_ok) / len(turnos_ok)
        avg_tps   = sum(m["tok_s"]     for m in turnos_ok) / len(turnos_ok)
        print(f"  Turnos exitosos: {len(turnos_ok)}/{N_TURNOS}")
        print(f"  Latencia STT promedio:   {avg_stt:.2f}s")
        print(f"  Latencia LLM promedio:   {avg_llm:.2f}s  ({avg_tps:.1f} tok/s avg)")
        print(f"  Latencia TOTAL promedio: {avg_total:.2f}s")
        print(f"  {'✅ Pipeline estable' if len(turnos_ok) == N_TURNOS else '⚠ Hubo fallos — revisar logs'}")
    else:
        print("  ❌ Ningún turno completó el ciclo completo.")
    print(f"{'═'*58}\n")


if __name__ == "__main__":
    asyncio.run(main())