"""
llm/llm_test.py
Proyecto Antonia — EAFIT
Prueba del módulo LLM con Ollama en Docker (Qwen 1.5B).
"""

import httpx
import time
import os
import subprocess

# ══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════

OLLAMA_URL = "http://localhost:11434/api/chat"

# El nuevo cerebro de Antonia
MODEL = "qwen2.5:1.5b"

# Ruta absoluta al system prompt (independiente de desde dónde se ejecute)
DIR_ACTUAL  = os.path.dirname(os.path.abspath(__file__))
RUTA_PROMPT = os.path.join(DIR_ACTUAL, "..", "config", "system_prompt.txt")


# ══════════════════════════════════════════════════════════════
# DIAGNÓSTICO DE ENTORNO
# ══════════════════════════════════════════════════════════════

def check_environment() -> bool:
    """Verifica que Ollama está corriendo en Docker y el modelo existe."""

    # 1. ¿Está el servidor Ollama respondiendo?
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        r.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Ollama no responde en localhost:11434 → {e}")
        print("        Verifica que el contenedor esté arriba con: sudo docker ps")
        print("        Ejecuta: sudo docker start ollama_antonia")
        return False

    # 2. ¿Existe el modelo solicitado?
    modelos_disponibles = [m["name"] for m in r.json().get("models", [])]
    if not modelos_disponibles:
        print("[ERROR] No hay modelos descargados.")
        print(f"        Ejecuta: sudo docker exec -it ollama_antonia ollama pull {MODEL}")
        return False

    # Buscar coincidencia parcial si el nombre exacto no está
    match = next((m for m in modelos_disponibles if MODEL.split(":")[0] in m), None)
    if MODEL not in modelos_disponibles:
        print(f"[WARN]  Modelo '{MODEL}' no encontrado.")
        print(f"        Modelos disponibles: {modelos_disponibles}")
        if match:
            print(f"        Usando '{match}' en su lugar.")
            globals()["MODEL"] = match
        else:
            return False

    # 3. Memoria RAM disponible (advertencia si hay menos de 3GB libres)
    try:
        mem = subprocess.check_output(["free", "-m"]).decode()
        libre_mb = int(mem.split("\n")[1].split()[3])
        if libre_mb < 3000:
            print(f"[WARN]  RAM libre: {libre_mb}MB — puede haber OOM.")
            print("        Cierra otros procesos antes de la prueba.")
    except Exception:
        pass

    print(f"[OK]    Ollama activo en Docker. Usando modelo: {MODEL}\n")
    return True


# ══════════════════════════════════════════════════════════════
# CARGA DEL SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

def load_system_prompt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        print(f"[OK]    System prompt cargado ({len(content)} chars)\n")
        return content
    except FileNotFoundError:
        print(f"[ERROR] No se encontró el system prompt en: {path}")
        print("        Crea el archivo config/system_prompt.txt temporalmente para la prueba.")
        # Si no existe, devolvemos un prompt de emergencia para que la prueba no falle
        return "Eres Antonia, la asistente del Laboratorio de Control Digital de EAFIT. Responde de forma técnica, breve y en español."


# ══════════════════════════════════════════════════════════════
# CLIENTE LLM
# ══════════════════════════════════════════════════════════════

def ask(question: str, system: str, history: list[dict] | None = None) -> str:
    """Envía una pregunta a Ollama y devuelve la respuesta como string."""
    
    messages = [{"role": "system", "content": system}]

    if history:
        messages.extend(history[-6:])  # Últimos 3 turnos (6 mensajes) — evita OOM

    messages.append({"role": "user", "content": question})

    payload = {
        "model": MODEL,
        "stream": False,  # <-- CORRECCIÓN: False para poder leer r.json() de un solo golpe
        "options": {
            "num_ctx": 1024,        # Reducido a 1024 para alinearse con nuestro Setup Seguro
            "temperature": 0.2,     # Más determinista
            "top_p": 0.9,
            "repeat_penalty": 1.1,  
            "num_predict": 150,     # Máximo de tokens en respuesta (~3-4 oraciones)
        },
        "messages": messages,
    }

    try:
        t0 = time.time()
        r  = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
        r.raise_for_status()
        data    = r.json()
        latency = time.time() - t0

        respuesta = data.get("message", {}).get("content", "").strip()
        tokens_ps = data.get("eval_count", 0) / max(data.get("eval_duration", 1), 1) * 1e9

        print(f"  [LLM] Latencia: {latency:.2f}s | "
              f"Tokens: {data.get('eval_count', '?')} | "
              f"Velocidad: {tokens_ps:.1f} tok/s")

        return respuesta

    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        if "runner process has terminated" in error_body:
            return (
                "[ERROR CRÍTICO] Ollama crasheó. Causas más probables:\n"
                "  1. OOM de GPU — RAM física agotada.\n"
                "  2. Revisa tegrastats para confirmar el uso de memoria."
            )
        return f"[ERROR HTTP] {error_body}"

    except httpx.TimeoutException:
        return "[ERROR] Timeout (>60s). El modelo tarda demasiado en cargar."

    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════
# MAIN — PRUEBAS
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # 1. Verificar entorno
    if not check_environment():
        raise SystemExit(1)

    # 2. Cargar system prompt
    SYSTEM = load_system_prompt(RUTA_PROMPT)

    # 3. Pruebas
    historial: list[dict] = []

    preguntas = [
        "¿Cuál es el horario del laboratorio?",
        "¿Dónde están los multímetros?",
        "¿Cómo conecto un PLC Siemens S7-1200?",
        "¿Cuáles son las normas de seguridad básicas?",
        "¿Cuántos núcleos tiene el procesador del Arduino Due?",
    ]

    for pregunta in preguntas:
        print(f"\n{'─'*52}")
        print(f"  PREGUNTA  : {pregunta}")
        respuesta = ask(pregunta, SYSTEM, historial)
        print(f"  RESPUESTA : {respuesta}")

        # Acumular historial
        historial.append({"role": "user",      "content": pregunta})
        historial.append({"role": "assistant", "content": respuesta})

    print(f"\n{'─'*52}")
    print("  Prueba completada.")