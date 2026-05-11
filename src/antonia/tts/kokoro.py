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

        from kokoro_onnx import Kokoro

        providers = onnx_providers or ["CPUExecutionProvider"]
        try:
            self._kokoro = Kokoro(str(model_path), str(voices_path), providers=providers)
            log.info("kokoro_loaded", providers=providers)
        except TypeError:
            self._kokoro = Kokoro(str(model_path), str(voices_path))
            log.warning("kokoro_no_providers_kwarg", fallback="CPU")

    def synthesize(
        self, text: str, lang: str | None = None
    ) -> tuple[npt.NDArray[np.float32], int]:
        lang = lang or _detect_language(text)
        voice = VOICE_ES if lang == "es" else VOICE_EN
        lang_code = "es" if lang == "es" else "en-us"
        samples, sr = self._kokoro.create(text, voice=voice, speed=1.0, lang=lang_code)
        return np.array(samples, dtype=np.float32), int(sr)
