"""
src/antonia/tts/piper.py

PiperBackend — warm-pool subprocess for Piper TTS.
R-3: Replacement process spawned immediately after communicate() to hide ELF loader cost.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import structlog

log = structlog.get_logger(__name__)


class PiperBackend:
    def __init__(self, es_model: Path, en_model: Path) -> None:
        self._es = es_model if es_model.exists() else None
        self._en = en_model if en_model.exists() else None

        if not self._es and not self._en:
            raise FileNotFoundError("Piper: no model files found")

        if not self._es:
            log.warning("piper_es_missing")
        if not self._en:
            log.warning("piper_en_missing")

        self._warm: dict[str, Optional[subprocess.Popen[bytes]]] = {"es": None, "en": None}
        if self._es:
            self._warm["es"] = self._spawn("es")

    def _model_for(self, lang: str) -> Optional[Path]:
        if lang == "en" and self._en:
            return self._en
        return self._es or self._en

    def _spawn(self, lang: str) -> Optional[subprocess.Popen[bytes]]:
        model = self._model_for(lang)
        if model is None:
            return None
        return subprocess.Popen(
            ["piper", "--model", str(model), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def synthesize(
        self, text: str, lang: str = "es"
    ) -> tuple[npt.NDArray[np.float32], int]:
        proc = self._warm.get(lang)
        if proc is None or proc.poll() is not None:
            proc = self._spawn(lang)

        if proc is None:
            raise RuntimeError(f"Piper: no model for lang={lang}")

        try:
            raw, _ = proc.communicate(input=text.encode("utf-8"), timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            raw, _ = proc.communicate()
        except Exception as exc:
            log.warning("piper_communicate_failed", error=str(exc))
            raw = b""

        self._warm[lang] = self._spawn(lang)

        if not raw:
            raise RuntimeError("Piper produced no audio")

        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return audio, 22050
