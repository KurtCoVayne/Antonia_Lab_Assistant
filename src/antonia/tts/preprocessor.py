"""
src/antonia/tts/preprocessor.py

TextPreprocessor — cleans LLM output before synthesis.
Phonetic map is loaded from knowledge_base/phonetic_map.yaml and hot-reloadable.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

import structlog
import yaml

log = structlog.get_logger(__name__)

# Class-level compiled unit substitutions — built once at class definition time.
_UNIT_SUBS_RAW = [
    (r"\b(\d+(?:\.\d+)?)\s*°C\b", r"\1 grados Celsius"),
    (r"\b(\d+(?:\.\d+)?)\s*°F\b", r"\1 grados Fahrenheit"),
    (r"\b(\d+(?:\.\d+)?)\s*kHz\b", r"\1 kilohercios"),
    (r"\b(\d+(?:\.\d+)?)\s*MHz\b", r"\1 megahercios"),
    (r"\b(\d+(?:\.\d+)?)\s*GHz\b", r"\1 gigahercios"),
    (r"\b(\d+(?:\.\d+)?)\s*GB\b",  r"\1 gigabytes"),
    (r"\b(\d+(?:\.\d+)?)\s*MB\b",  r"\1 megabytes"),
    (r"\b(\d+(?:\.\d+)?)\s*ms\b",  r"\1 milisegundos"),
    (r"\b(\d+(?:\.\d+)?)\s*V\b",   r"\1 voltios"),
    (r"\b(\d+(?:\.\d+)?)\s*mA\b",  r"\1 miliamperios"),
    (r"\b(\d+(?:\.\d+)?)\s*W\b",   r"\1 vatios"),
    (r"\b(\d+(?:\.\d+)?)\s*%\b",   r"\1 por ciento"),
]
_COMPILED_UNIT_SUBS = [(re.compile(p, re.IGNORECASE), r) for p, r in _UNIT_SUBS_RAW]


class TextPreprocessor:
    def __init__(self, phonetic_map_path: Path) -> None:
        self._phonetic_path = phonetic_map_path
        self._compiled_phonetics: list[tuple[re.Pattern[str], str]] = []
        self.load_phonetic_map()

    def load_phonetic_map(self) -> None:
        """
        Load and compile phonetic map. Call after RAG ingestion updates the YAML.
        Keys starting with '_' are ignored (comments/metadata).
        """
        if not self._phonetic_path.exists():
            self._bootstrap_phonetic_map()

        raw_map: dict[str, str] = {}
        try:
            with self._phonetic_path.open(encoding="utf-8") as f:
                data: dict[str, Any] = yaml.safe_load(f)
            terms = data.get("terms", {}) if isinstance(data, dict) else data
            raw_map = {k: v for k, v in terms.items() if not str(k).startswith("_")}
            log.info("phonetic_map_loaded", count=len(raw_map))
        except Exception as exc:
            log.warning("phonetic_map_load_error", error=str(exc))

        compiled = []
        for key, phonetic in raw_map.items():
            try:
                compiled.append((re.compile(key, re.IGNORECASE), str(phonetic)))
            except re.error as exc:
                log.warning("phonetic_map_invalid_pattern", key=key, error=str(exc))
        self._compiled_phonetics = compiled

    def _bootstrap_phonetic_map(self) -> None:
        self._phonetic_path.parent.mkdir(parents=True, exist_ok=True)
        base: dict[str, Any] = {
            "version": "1.1",
            "managed_by": "ingest_pipeline",
            "terms": {
                "\\bPLC\\b":          "Pe-ele-ce",
                "\\bPLCs\\b":         "Pe-ele-ces",
                "\\bHMI\\b":          "Hache-eme-i",
                "\\bIoT\\b":          "I-o-Te",
                "\\bCPU\\b":          "Ce-pe-u",
                "\\bGPU\\b":          "Ge-pe-u",
                "\\bUSB\\b":          "U-ese-be",
                "\\bHDMI\\b":         "Hache-de-eme-i",
                "\\bLED\\b":          "led",
                "\\bLEDs\\b":         "leds",
                "\\bPWM\\b":          "Pe-doble-uve-eme",
                "\\bI2C\\b":          "I-dos-ce",
                "\\bSPI\\b":          "ese-pe-i",
                "\\bUART\\b":         "U-a-erre-te",
                "\\bFPGA\\b":         "Fe-pe-ge-a",
                "\\bJetson\\b":       "Yetson",
                "\\bNVIDIA\\b":       "En-vidia",
                "\\bSiemens\\b":      "Síemens",
                "\\bArduino\\b":      "Arduíno",
                "\\bRaspberry Pi\\b": "Ráspberri Pai",
                "\\bLabVIEW\\b":      "Lab-viu",
                "\\bMATLAB\\b":       "Matlab",
                "\\bPython\\b":       "Páiton",
                "\\bGitHub\\b":       "Git-jab",
                "\\bWi-Fi\\b":        "Güái-fai",
                "\\bBluetooth\\b":    "Blú-tuz",
                "\\bEthernet\\b":     "Éternet",
                "\\bJSON\\b":         "Yéison",
                "\\bAPI\\b":          "A-pe-i",
                "\\bEAFIT\\b":        "E-a-fit",
                "\\bRAM\\b":          "ram",
                "\\bSSD\\b":          "ese-ese-de",
                "\\bTIA Portal\\b":   "Tía Portal",
                "\\bS7-1200\\b":      "ese-siete doce-cero-cero",
                "\\bPROFINET\\b":     "Pro-fi-net",
            },
        }
        try:
            with self._phonetic_path.open("w", encoding="utf-8") as f:
                yaml.dump(base, f, allow_unicode=True, default_flow_style=False)
            log.info("phonetic_map_bootstrapped", path=str(self._phonetic_path))
        except Exception as exc:
            log.warning("phonetic_map_bootstrap_error", error=str(exc))

    def process(self, text: str) -> str:
        text = self._strip_markdown(text)
        text = self._expand_units(text)
        text = self._apply_phonetics(text)
        text = self._fix_punctuation(text)
        return self._clean(text)

    def _strip_markdown(self, text: str) -> str:
        text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text, flags=re.DOTALL)
        text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"https?://\S+", "el enlace", text)
        text = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U00002600-\U000026FF]",
            "",
            text,
        )
        return text

    def _expand_units(self, text: str) -> str:
        for pattern, repl in _COMPILED_UNIT_SUBS:
            text = pattern.sub(repl, text)
        return text

    def _apply_phonetics(self, text: str) -> str:
        for pattern, phonetic in self._compiled_phonetics:
            text = pattern.sub(phonetic, text)
        return text

    def _fix_punctuation(self, text: str) -> str:
        text = re.sub(r"\n{2,}", ". ", text)
        text = re.sub(r"\n", ", ", text)
        text = re.sub(r"\.{2,}", ".", text)
        text = re.sub(r"([.!?])\s*([.!?])+", r"\1", text)
        return text

    def _clean(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = "".join(c for c in text if unicodedata.category(c) not in ("Cc", "Cf"))
        return text.strip()
