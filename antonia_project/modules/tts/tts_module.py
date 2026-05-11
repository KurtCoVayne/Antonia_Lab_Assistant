"""
modules/tts/tts_module.py
Proyecto Antonia — EAFIT

Pipeline TTS:
  TextPreprocessor  → limpia LLM output + fonética desde phonetic_map.json
  LanguageDetector  → detecta ES/EN
  KokoroEngine      → motor primario CPU/GPU (ef_dora ES, af_bella EN)
  PiperEngine       → fallback con proceso persistente warm-pool

FONÉTICA:
  No está hardcodeada aquí.
  Vive en knowledge_base/phonetic_map.json.
  Se crea automáticamente en la primera ejecución.
  El pipeline RAG la enriquecerá sin tocar este módulo.
  Ver PUERTA_RAG en TextPreprocessor.load_phonetic_map().
"""

import os
import re
import json
import time
import subprocess
import unicodedata
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf

# ── Rutas ──────────────────────────────────────────────────────────────────
BASE_DIR = "/media/antonia_ssd/antonia/antonia_project"

KOKORO_MODEL      = os.path.join(BASE_DIR, "models/kokoro/kokoro-v1.0.onnx")
KOKORO_VOICES     = os.path.join(BASE_DIR, "models/kokoro/voices-v1.0.bin")
PIPER_ES_MODEL    = os.path.join(BASE_DIR, "models/piper/es_MX-claude-high.onnx")
PIPER_EN_MODEL    = os.path.join(BASE_DIR, "models/piper/en_US-lessac-medium.onnx")
PHONETIC_MAP_PATH = os.path.join(BASE_DIR, "knowledge_base/phonetic_map.json")
OUTPUT_DIR        = os.path.join(BASE_DIR, "tests")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# CAPA 1 — PREPROCESADOR
# ══════════════════════════════════════════════════════════════

class TextPreprocessor:
    """
    Limpia el output del LLM antes de enviarlo al motor TTS.
    La fonética técnica se carga desde un archivo JSON externo.
    """

    _UNIT_SUBS = [
        (r'\b(\d+(?:\.\d+)?)\s*°C\b',  r'\1 grados Celsius'),
        (r'\b(\d+(?:\.\d+)?)\s*°F\b',  r'\1 grados Fahrenheit'),
        (r'\b(\d+(?:\.\d+)?)\s*kHz\b', r'\1 kilohercios'),
        (r'\b(\d+(?:\.\d+)?)\s*MHz\b', r'\1 megahercios'),
        (r'\b(\d+(?:\.\d+)?)\s*GHz\b', r'\1 gigahercios'),
        (r'\b(\d+(?:\.\d+)?)\s*GB\b',  r'\1 gigabytes'),
        (r'\b(\d+(?:\.\d+)?)\s*MB\b',  r'\1 megabytes'),
        (r'\b(\d+(?:\.\d+)?)\s*ms\b',  r'\1 milisegundos'),
        (r'\b(\d+(?:\.\d+)?)\s*V\b',   r'\1 voltios'),
        (r'\b(\d+(?:\.\d+)?)\s*mA\b',  r'\1 miliamperios'),
        (r'\b(\d+(?:\.\d+)?)\s*W\b',   r'\1 vatios'),
        (r'\b(\d+(?:\.\d+)?)\s*%\b',   r'\1 por ciento'),
    ]
    # R-2: compilar unit subs una sola vez al definir la clase
    _COMPILED_UNIT_SUBS = [
        (re.compile(p, re.IGNORECASE), r) for p, r in _UNIT_SUBS
    ]

    def __init__(self):
        # R-2: lista de (Pattern compilado, reemplazo) actualizada en load_phonetic_map()
        self._compiled_phonetics: list[tuple[re.Pattern, str]] = []
        self.load_phonetic_map()

    # ── PUERTA_RAG ────────────────────────────────────────────────────────
    def load_phonetic_map(self) -> None:
        """
        Carga knowledge_base/phonetic_map.json en memoria y compila los patrones.

        PUERTA_RAG — integración futura con el pipeline RAG:
          Cuando ingest_kb.py indexe nuevos documentos del laboratorio,
          deberá:
            1. Extraer términos técnicos nuevos (siglas, modelos de equipos)
            2. Añadirlos a phonetic_map.json con su pronunciación fonética
            3. Llamar tts.preprocessor.load_phonetic_map() para recarga en caliente
          Así el TTS pronuncia términos nuevos sin reiniciar el sistema.

          Formato del JSON:
            { "\\bPLC\\b": "Pe-ele-ce", "\\bJetson\\b": "Yetson", ... }
            Las claves son patrones regex case-insensitive.
            Las claves que empiezan con "_" son comentarios y se ignoran.
        """
        if not os.path.exists(PHONETIC_MAP_PATH):
            self._bootstrap_phonetic_map()

        raw_map: dict[str, str] = {}
        try:
            with open(PHONETIC_MAP_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw_map = {k: v for k, v in raw.items() if not k.startswith("_")}
            print(f"[TTS]  Fonética: {len(raw_map)} términos cargados")
        except Exception as e:
            print(f"[TTS]  ⚠ phonetic_map.json no se pudo cargar: {e}")

        # R-2: compilar patrones en load_phonetic_map(), no en cada speak().
        # Los patrones inválidos se registran con la clave exacta para diagnóstico.
        compiled = []
        for key, phonetic in raw_map.items():
            try:
                compiled.append((re.compile(key, re.IGNORECASE), phonetic))
            except re.error as exc:
                print(f"[TTS]  ⚠ Patrón inválido en phonetic_map.json (clave='{key}'): {exc}")
        self._compiled_phonetics = compiled

    def _bootstrap_phonetic_map(self) -> None:
        os.makedirs(os.path.dirname(PHONETIC_MAP_PATH), exist_ok=True)

        base = {
            "_comment": "Fonética técnica de Antonia. Claves: regex case-insensitive. Enriquecido por RAG.",
            "_version": "1.0",
            "\\bPLC\\b":            "Pe-ele-ce",
            "\\bPLCs\\b":           "Pe-ele-ces",
            "\\bHMI\\b":            "Hache-eme-i",
            "\\bIoT\\b":            "I-o-Te",
            "\\bCPU\\b":            "Ce-pe-u",
            "\\bGPU\\b":            "Ge-pe-u",
            "\\bUSB\\b":            "U-ese-be",
            "\\bHDMI\\b":           "Hache-de-eme-i",
            "\\bLED\\b":            "led",
            "\\bLEDs\\b":           "leds",
            "\\bPWM\\b":            "Pe-doble-uve-eme",
            "\\bI2C\\b":            "I-dos-ce",
            "\\bSPI\\b":            "ese-pe-i",
            "\\bUART\\b":           "U-a-erre-te",
            "\\bFPGA\\b":           "Fe-pe-ge-a",
            "\\bJetson\\b":         "Yetson",
            "\\bNVIDIA\\b":         "En-vidia",
            "\\bSiemens\\b":        "Síemens",
            "\\bArduino\\b":        "Arduíno",
            "\\bRaspberry Pi\\b":   "Ráspberri Pai",
            "\\bLabVIEW\\b":        "Lab-viu",
            "\\bMATLAB\\b":         "Matlab",
            "\\bPython\\b":         "Páiton",
            "\\bGitHub\\b":         "Git-jab",
            "\\bWi-Fi\\b":          "Güái-fai",
            "\\bBluetooth\\b":      "Blú-tuz",
            "\\bEthernet\\b":       "Éternet",
            "\\bJSON\\b":           "Yéison",
            "\\bAPI\\b":            "A-pe-i",
            "\\bEAFIT\\b":          "E-a-fit",
            "\\bRAM\\b":            "ram",
            "\\bSSD\\b":            "ese-ese-de",
            "\\bTIA Portal\\b":     "Tía Portal",
            "\\bS7-1200\\b":        "ese-siete doce-cero-cero",
            "\\bPROFINET\\b":       "Pro-fi-net",
        }

        try:
            with open(PHONETIC_MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(base, f, ensure_ascii=False, indent=2)
            print(f"[TTS]  phonetic_map.json creado con {len(base)-2} términos base")
        except Exception as e:
            print(f"[TTS]  ⚠ No se pudo crear phonetic_map.json: {e}")

    # ── Pipeline de normalización ─────────────────────────────────────────

    def process(self, text: str) -> str:
        text = self._strip_markdown(text)
        text = self._expand_units(text)
        text = self._apply_phonetics(text)
        text = self._fix_punctuation(text)
        text = self._clean(text)
        return text

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text, flags=re.DOTALL)
        text = re.sub(r'_{1,3}(.+?)_{1,3}',   r'\1', text, flags=re.DOTALL)
        text = re.sub(r'`{1,3}[^`]*`{1,3}', '', text)
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'https?://\S+', 'el enlace', text)
        text = re.sub(
            r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF'
            r'\U0001F680-\U0001F6FF\U00002600-\U000026FF]', '', text
        )
        return text

    def _expand_units(self, text: str) -> str:
        # R-2: usar patrones pre-compilados a nivel de clase
        for pattern, repl in self._COMPILED_UNIT_SUBS:
            text = pattern.sub(repl, text)
        return text

    def _apply_phonetics(self, text: str) -> str:
        # R-2: patrones pre-compilados en load_phonetic_map(), cero compilación aquí
        for pattern, phonetic in self._compiled_phonetics:
            text = pattern.sub(phonetic, text)
        return text

    def _fix_punctuation(self, text: str) -> str:
        text = re.sub(r'\n{2,}', '. ', text)
        text = re.sub(r'\n',     ', ', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'([.!?])\s*([.!?])+', r'\1', text)
        return text

    def _clean(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = ''.join(
            c for c in text
            if unicodedata.category(c) not in ('Cc', 'Cf')
        )
        return text.strip()


# ══════════════════════════════════════════════════════════════
# CAPA 2 — DETECTOR DE IDIOMA
# ══════════════════════════════════════════════════════════════

class LanguageDetector:
    """
    Detecta si el texto es ES o EN.
    Usa langdetect si está disponible; cae a heurística de palabras comunes.
    """

    _EN_WORDS = frozenset([
        "the","is","are","was","were","have","has","do","does","can","will",
        "would","could","should","and","or","but","not","with","from","this",
        "that","your","you","we","they","it","be","been","please","hello",
        "hi","how","what","where","when","why","all","some","any",
    ])

    def __init__(self):
        self._use_langdetect = False
        try:
            from langdetect import detect, DetectorFactory
            DetectorFactory.seed = 42
            self._detect_fn      = detect
            self._use_langdetect = True
            print("[TTS]  langdetect disponible ✅")
        except ImportError:
            print("[TTS]  langdetect no instalado — heurística activa "
                  "(pip install langdetect para mayor precisión)")

    def detect(self, text: str) -> str:
        if len(text.strip()) < 10:
            return "es"
        if self._use_langdetect:
            try:
                lang = self._detect_fn(text)
                return lang if lang in ("es", "en") else "es"
            except Exception:
                pass
        words    = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        en_count = sum(1 for w in words if w in self._EN_WORDS)
        return "en" if words and (en_count / len(words)) > 0.30 else "es"


# ══════════════════════════════════════════════════════════════
# CAPA 3 — MOTOR KOKORO (PRIMARIO)
# ══════════════════════════════════════════════════════════════

class KokoroEngine:
    """
    Kokoro-82M ONNX.
    ef_dora → femenina latinoamericana (contexto EAFIT Colombia)
    af_bella → inglés americano

    H-3: Intenta CUDAExecutionProvider primero (GPU libre durante TTS en el
    sistema de relevos). Cae a CPU si el proveedor CUDA no está disponible o
    si onnxruntime-gpu no está instalado.
    IMPORTANTE: solo activar H-3 si C-2 (OLLAMA_KEEP_ALIVE_S > 0) está
    configurado correctamente; de lo contrario Qwen puede seguir en VRAM.
    """

    VOICE_ES = "ef_dora"
    VOICE_EN = "af_bella"

    def __init__(self, model_path: str, voices_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Kokoro model: {model_path}")
        if not os.path.exists(voices_path):
            raise FileNotFoundError(f"Kokoro voices: {voices_path}")

        from kokoro_onnx import Kokoro

        # H-3: preferir GPU si onnxruntime-gpu está instalado y la VRAM está libre
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        try:
            self._kokoro = Kokoro(model_path, voices_path, providers=providers)
            print("[TTS]  Kokoro: intentando CUDAExecutionProvider")
        except TypeError:
            # kokoro_onnx version que no acepta providers → CPU
            self._kokoro = Kokoro(model_path, voices_path)
            print("[TTS]  ⚠ kokoro_onnx no acepta providers — usando CPU")

    def synthesize(self, text: str, lang: str = "es") -> tuple[np.ndarray, int]:
        voice     = self.VOICE_ES if lang == "es" else self.VOICE_EN
        lang_code = "es"          if lang == "es" else "en-us"
        samples, sr = self._kokoro.create(text, voice=voice, speed=1.0, lang=lang_code)
        return np.array(samples, dtype=np.float32), int(sr)


# ══════════════════════════════════════════════════════════════
# CAPA 4 — MOTOR PIPER (FALLBACK)
# ══════════════════════════════════════════════════════════════

class PiperEngine:
    """
    Fallback monolingüe con warm-pool de procesos (R-3).

    R-3: Mantiene un proceso Piper pre-calentado por idioma.
    Tras cada síntesis se reabre un proceso de reemplazo en paralelo con
    la reproducción de audio, ocultando el costo del ELF loader detrás
    del tiempo de playback. El proceso de reemplazo está listo cuando
    llega la siguiente llamada.
    """

    def __init__(self, es_model: str, en_model: str):
        self._es = es_model if os.path.exists(es_model) else None
        self._en = en_model if os.path.exists(en_model) else None
        if not self._es and not self._en:
            raise FileNotFoundError("Piper: ningún modelo encontrado")
        if not self._es:
            print("[TTS]  ⚠ Piper ES no encontrado")
        if not self._en:
            print("[TTS]  ⚠ Piper EN no encontrado")

        # R-3: procesos warm por idioma; None → se crea en la primera llamada
        self._warm: dict[str, Optional[subprocess.Popen]] = {"es": None, "en": None}
        # Pre-calentar el idioma primario (ES)
        if self._es:
            self._warm["es"] = self._spawn("es")

    def _model_for(self, lang: str) -> Optional[str]:
        return (self._en if lang == "en" and self._en else None) or self._es or self._en

    def _spawn(self, lang: str) -> Optional[subprocess.Popen]:
        model = self._model_for(lang)
        if model is None:
            return None
        return subprocess.Popen(
            ["piper", "--model", model, "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def synthesize(self, text: str, lang: str = "es") -> tuple[np.ndarray, int]:
        proc = self._warm.get(lang)
        if proc is None or proc.poll() is not None:
            proc = self._spawn(lang)

        try:
            raw, _ = proc.communicate(
                input=text.encode("utf-8"),
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            raw, _ = proc.communicate()
        except Exception as e:
            print(f"[TTS]  ⚠ Piper communicate falló ({e}) — reintentando")
            raw = b""

        # R-3: spawn del proceso de reemplazo en background antes de convertir audio
        # El proceso estará listo cuando termine la reproducción de este turno.
        self._warm[lang] = self._spawn(lang)

        if not raw:
            raise RuntimeError("Piper no produjo audio")

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, 22050


# ══════════════════════════════════════════════════════════════
# MÓDULO PRINCIPAL
# ══════════════════════════════════════════════════════════════

class AntoniaTTS:
    """
    Motor TTS completo de Antonia.

    Desde el pipeline / orquestador:
        from modules.tts.tts_module import tts
        tts.speak(respuesta_llm)
        tts.speak(texto, save_wav=True, wav_filename="turno_1.wav")
        tts.preprocessor.load_phonetic_map()   # recarga tras actualizar RAG
    """

    def __init__(self):
        print("[TTS]  Inicializando...")
        self.preprocessor    = TextPreprocessor()
        self.detector        = LanguageDetector()
        self._kokoro: Optional[KokoroEngine] = None
        self._piper:  Optional[PiperEngine]  = None
        self._engine_name    = None
        self._load_engines()
        print(f"[TTS]  ✅ Motor activo: {self._engine_name}\n")

    def _load_engines(self) -> None:
        try:
            self._kokoro      = KokoroEngine(KOKORO_MODEL, KOKORO_VOICES)
            self._engine_name = "kokoro"
            print("[TTS]  ✅ Kokoro-82M cargado")
        except Exception as e:
            print(f"[TTS]  ⚠ Kokoro no disponible: {e}")

        try:
            self._piper = PiperEngine(PIPER_ES_MODEL, PIPER_EN_MODEL)
            if self._engine_name is None:
                self._engine_name = "piper"
            print("[TTS]  ✅ Piper cargado como fallback (warm-pool)")
        except Exception as e:
            print(f"[TTS]  ⚠ Piper no disponible: {e}")

        if self._engine_name is None:
            raise RuntimeError(
                "[TTS] CRÍTICO: Ningún motor TTS disponible. "
                "Verifica modelos en models/kokoro/ y models/piper/"
            )

    def speak(
        self,
        text: str,
        save_wav: bool = False,
        wav_filename: str = "antonia_output.wav",
        play_audio: bool = True,
    ) -> Optional[str]:
        """
        Sintetiza y reproduce texto.
        Acepta el output crudo del LLM — el preprocesador limpia el markdown.
        """
        if not text or not text.strip():
            print("[TTS]  ⚠ Texto vacío.")
            return None

        t0 = time.time()

        processed = self.preprocessor.process(text)
        if not processed:
            return None

        lang        = self.detector.detect(processed)
        samples, sr = self._synthesize(processed, lang)

        if samples is None:
            print("[TTS]  ❌ Síntesis fallida.")
            return None

        if play_audio:
            try:
                sd.play(samples, sr)
                sd.wait()
            except Exception as e:
                print(f"[TTS]  ⚠ Error al reproducir: {e}")

        filepath = None
        if save_wav:
            filepath = os.path.join(OUTPUT_DIR, wav_filename)
            try:
                sf.write(filepath, samples, sr)
                print(f"[TTS]  💾 WAV guardado: {filepath}")
            except Exception as e:
                print(f"[TTS]  ⚠ No se pudo guardar WAV: {e}")

        print(f"[TTS]  🔊 {self._engine_name} | {lang.upper()} | "
              f"{time.time() - t0:.2f}s | '{processed[:55]}...'")
        return filepath

    def _synthesize(self, text: str, lang: str) -> tuple[Optional[np.ndarray], int]:
        if self._kokoro:
            try:
                return self._kokoro.synthesize(text, lang)
            except Exception as e:
                print(f"[TTS]  ⚠ Kokoro falló ({e}) — usando Piper")
        if self._piper:
            try:
                self._engine_name = "piper"
                return self._piper.synthesize(text, lang)
            except Exception as e:
                print(f"[TTS]  ❌ Piper también falló: {e}")
        return None, 0


# ── Instancia global ───────────────────────────────────────────────────────
try:
    tts = AntoniaTTS()
except RuntimeError as e:
    print(f"\n{e}\n")
    tts = None
