"""
src/antonia/tts/kokoro.py

KokoroBackend — Kokoro-82M ONNX synthesis.
H-3: Tries CUDAExecutionProvider first; falls back to CPU silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import structlog

log = structlog.get_logger(__name__)

_EN_WORDS = frozenset([
    "the", "is", "are", "was", "were", "have", "has", "do", "does", "can",
    "will", "would", "could", "should", "and", "or", "but", "not", "with",
    "from", "this", "that", "your", "you", "we", "they", "it", "be", "been",
    "please", "hello", "hi", "how", "what", "where", "when", "why",
])

VOICE_ES = "ef_dora"
VOICE_EN = "af_bella"


def _detect_language(text: str) -> str:
    if len(text.strip()) < 10:
        return "es"
    try:
        from langdetect import detect, DetectorFactory
        DetectorFactory.seed = 42
        lang = detect(text)
        return lang if lang in ("es", "en") else "es"
    except Exception:
        pass
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    en_count = sum(1 for w in words if w in _EN_WORDS)
    return "en" if words and (en_count / len(words)) > 0.30 else "es"


class KokoroBackend:
    def __init__(
        self,
        model_path: Path,
        voices_path: Path,
        onnx_providers: list[str] | None = None,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Kokoro model not found: {model_path}")
        if not voices_path.exists():
            raise FileNotFoundError(f"Kokoro voices not found: {voices_path}")

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        cpu_providers = ["CPUExecutionProvider"]
        preferred_providers = onnx_providers or cpu_providers

        cpu_session = ort.InferenceSession(str(model_path), providers=cpu_providers)
        self._kokoro_cpu = Kokoro.from_session(cpu_session, str(voices_path))
        log.info("kokoro_loaded", providers=cpu_providers)

        if preferred_providers != cpu_providers:
            try:
                cuda_session = ort.InferenceSession(str(model_path), providers=preferred_providers)
                self._kokoro_cuda = Kokoro.from_session(cuda_session, str(voices_path))
                log.info("kokoro_cuda_loaded", providers=preferred_providers)
            except Exception as exc:
                log.warning("kokoro_cuda_fallback", error=str(exc))
                self._kokoro_cuda = self._kokoro_cpu
        else:
            self._kokoro_cuda = self._kokoro_cpu

    def synthesize(
        self, text: str, lang: str | None = None, force_cpu: bool = False
    ) -> tuple[npt.NDArray[np.float32], int]:
        lang = lang or _detect_language(text)
        voice = VOICE_ES if lang == "es" else VOICE_EN
        lang_code = "es" if lang == "es" else "en-us"
        kokoro = self._kokoro_cpu if force_cpu else self._kokoro_cuda
        samples, sr = kokoro.create(text, voice=voice, speed=1.0, lang=lang_code)
        return np.array(samples, dtype=np.float32), int(sr)
